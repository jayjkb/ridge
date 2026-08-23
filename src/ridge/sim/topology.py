"""Fixed Stage-1 topology with its naming, addressing, FRRouting startup, and CSV export."""

from __future__ import annotations

import csv
import ipaddress
import os
import pwd
import shlex
import shutil
import time
from pathlib import Path
from typing import Any

from ridge.common.contracts import canonical_edge_id as canonical_edge_id

# WAN mesh profiles: host access is fast, metro/core links are mixed.
HOST_LINK = {"bw": 100, "delay": "0.10ms", "loss": 0.01, "use_tbf": True}
FABRIC_LINK = {"bw": 30, "delay": "1.6ms", "loss": 0.05, "use_tbf": True}
WAN_LINK = {"bw": 10, "delay": "7.5ms", "loss": 0.10, "use_tbf": True}
LOGICAL_NAME_KEY = "logical_name"
EDGE_ROLE_ACCESS = "access"
EDGE_ROLE_FABRIC = "fabric"
EDGE_ROLE_WAN = "wan"
HOST_SITE_A = "site_a"
HOST_SITE_B = "site_b"
HOST_SITE_UNKNOWN = "unknown"
HOST_SITE_BY_NAME = {
    "h1_1": HOST_SITE_A,
    "h1_2": HOST_SITE_A,
    "h1_3": HOST_SITE_A,
    "h2_1": HOST_SITE_A,
    "h2_2": HOST_SITE_A,
    "h2_3": HOST_SITE_A,
    "h3_1": HOST_SITE_A,
    "h3_2": HOST_SITE_A,
    "h3_3": HOST_SITE_A,
    "h4_1": HOST_SITE_B,
    "h4_2": HOST_SITE_B,
    "h5_1": HOST_SITE_B,
    "h5_2": HOST_SITE_B,
}


def _link(src: str, dst: str, profile: str, attrs: dict[str, object]) -> dict[str, object]:
    """Return one link specification with a copy of its profile attributes."""
    return {"src": src, "dst": dst, "profile": profile, "attrs": attrs.copy()}


def link_role_from_profile(profile: str) -> str:
    """Map a link profile name to its link tier."""
    if profile == "host":
        return EDGE_ROLE_ACCESS
    if profile == "fabric":
        return EDGE_ROLE_FABRIC
    if profile == "wan":
        return EDGE_ROLE_WAN
    raise ValueError(f"Unsupported link profile={profile}")


def edge_role_map() -> dict[str, str]:
    """Return the link tier of every link keyed by canonical link identifier."""
    roles: dict[str, str] = {}
    for link in topology_spec()["links"]:
        roles[canonical_edge_id(str(link["src"]), str(link["dst"]))] = link_role_from_profile(
            str(link["profile"])
        )
    return roles


def edge_role(edge_id: str) -> str:
    """Return the link tier of a link identifier given in either endpoint order."""
    roles = edge_role_map()
    normalized = edge_id
    if edge_id not in roles and "<->" in edge_id:
        src, dst = edge_id.split("<->", 1)
        normalized = canonical_edge_id(src, dst)
    if normalized not in roles:
        raise ValueError(f"Unknown edge_id={edge_id}")
    return roles[normalized]


def candidate_edge_ids_by_roles(roles: set[str] | None = None) -> list[str]:
    """Return the sorted link identifiers whose tier is in the requested set, or all tiers."""
    normalized = set(roles or {EDGE_ROLE_ACCESS, EDGE_ROLE_FABRIC, EDGE_ROLE_WAN})
    return sorted([edge_id for edge_id, role in edge_role_map().items() if role in normalized])


def topology_spec() -> dict[str, Any]:
    """Build the fixed 12-router, 13-host topology with its access, fabric, and wide-area links."""
    switches = [
        "r4",
        "r6",
        "r8",
        "r9",
        "r15",
        "r18",
        "r19",
        "r21",
        "r25",
        "r27",
        "r28",
        "r29",
    ]

    hosts = [
        "h1_1",
        "h1_2",
        "h1_3",
        "h2_1",
        "h2_2",
        "h2_3",
        "h3_1",
        "h3_2",
        "h3_3",
        "h4_1",
        "h4_2",
        "h5_1",
        "h5_2",
    ]

    links = []
    host_edges = {
        "r29": ["h1_1", "h1_2"],
        "r18": ["h1_3", "h2_1"],
        "r28": ["h2_2", "h2_3"],
        "r9": ["h3_1", "h3_2"],
        "r6": ["h3_3", "h4_1"],
        "r8": ["h4_2", "h5_1", "h5_2"],
    }
    for edge_router, edge_hosts in host_edges.items():
        links.extend(_link(host_name, edge_router, "host", HOST_LINK) for host_name in edge_hosts)

    # Router fabric that preserves the original access-side structure.
    fabric_edges = [
        ("r4", "r18"),
        ("r4", "r27"),
        ("r4", "r28"),
        ("r4", "r21"),
        ("r6", "r25"),
        ("r8", "r15"),
        ("r8", "r25"),
        ("r15", "r21"),
        ("r18", "r28"),
        ("r21", "r25"),
        ("r27", "r29"),
    ]
    links.extend(_link(src, dst, "fabric", FABRIC_LINK) for src, dst in fabric_edges)

    # WAN branch that keeps a slower perimeter route across the reduced graph.
    wan_edges = [
        ("r28", "r19"),
        ("r19", "r6"),
        ("r15", "r9"),
    ]
    links.extend(_link(src, dst, "wan", WAN_LINK) for src, dst in wan_edges)

    return {"switches": switches, "hosts": hosts, "links": links}


def node_role(name: str) -> str:
    """Return host for host names and spine for routers."""
    if name.startswith("h"):
        return "host"
    return "spine"


def host_site(name: str) -> str:
    """Return the site of a host, or unknown for other names."""
    return HOST_SITE_BY_NAME.get(name, HOST_SITE_UNKNOWN)


def runtime_name(logical_name: str, instance_id: str = "") -> str:
    """Prefix a logical node name with the Mininet instance identifier, checking the interface name limit."""
    if not instance_id:
        return logical_name
    if not instance_id.isalnum():
        raise ValueError("instance_id must contain only letters and digits")
    name = f"{instance_id}{logical_name}"
    if len(f"{name}-eth0") > 15:
        raise ValueError("instance_id is too long for Linux interface names")
    return name


def logical_name(node: object) -> str:
    """Return the logical name stored on a Mininet node, falling back to its runtime name."""
    return str(getattr(node, LOGICAL_NAME_KEY, getattr(node, "name")))


def logical_interface_name(node: object, interface: str) -> str:
    """Replace the runtime node prefix of an interface name with the logical name."""
    name = logical_name(node)
    runtime = str(getattr(node, "name"))
    if interface.startswith(f"{runtime}-"):
        return f"{name}{interface[len(runtime) :]}"
    return interface


def _runtime_interface_name(node: object, interface: str) -> str:
    """Replace the logical node prefix of an interface name with the runtime name."""
    logical = logical_name(node)
    runtime = str(getattr(node, "name"))
    if interface.startswith(f"{logical}-"):
        return f"{runtime}{interface[len(logical) :]}"
    return interface


def logical_node(net: object, name: str) -> object:
    """Return the Mininet node for a logical name within the network's instance."""
    return net.get(runtime_name(name, str(getattr(net, "instance_id", ""))))


def export_topology_csv(log_dir: Path, net: object | None = None) -> tuple[Path, Path]:
    """Write the topology node and link tables, with interfaces and addresses when a network is given."""
    spec = topology_spec()
    log_dir.mkdir(parents=True, exist_ok=True)
    node_path = log_dir / "topology_nodes.csv"
    link_path = log_dir / "topology_links.csv"

    with node_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Node", "NodeType", "Role"])
        writer.writeheader()
        writer.writerows(
            [
                {"Node": name, "NodeType": "router", "Role": node_role(name)}
                for name in spec["switches"]
            ]
            + [{"Node": name, "NodeType": "host", "Role": "host"} for name in spec["hosts"]]
        )

    interface_map: dict[str, tuple[str, str]] = {}
    if net is not None:
        for link in spec["links"]:
            src_node = logical_node(net, str(link["src"]))
            dst_node = logical_node(net, str(link["dst"]))
            src_intf, dst_intf = src_node.connectionsTo(dst_node)[0]
            edge_id = canonical_edge_id(str(link["src"]), str(link["dst"]))
            interface_map[edge_id] = (
                logical_interface_name(src_node, src_intf.name),
                logical_interface_name(dst_node, dst_intf.name),
            )

    with link_path.open("w", newline="") as handle:
        if net is None:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "Source",
                    "Destination",
                    "Profile",
                    "BandwidthMbps",
                    "Delay",
                    "LossPercent",
                ],
            )
            writer.writeheader()
            writer.writerows(
                {
                    "Source": link["src"],
                    "Destination": link["dst"],
                    "Profile": link["profile"],
                    "BandwidthMbps": link["attrs"]["bw"],
                    "Delay": link["attrs"]["delay"],
                    "LossPercent": link["attrs"]["loss"],
                }
                for link in spec["links"]
            )
        else:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "EdgeId",
                    "Source",
                    "Destination",
                    "SourceInterface",
                    "DestinationInterface",
                    "SourceIP",
                    "DestinationIP",
                    "Profile",
                    "BandwidthMbps",
                    "Delay",
                    "LossPercent",
                ],
            )
            writer.writeheader()
            for link in spec["links"]:
                edge_id = canonical_edge_id(str(link["src"]), str(link["dst"]))
                source_interface, destination_interface = interface_map.get(edge_id, ("", ""))
                src_ip = ""
                dst_ip = ""
                if source_interface and destination_interface:
                    src_node = logical_node(net, str(link["src"]))
                    dst_node = logical_node(net, str(link["dst"]))
                    src_runtime_intf = _runtime_interface_name(
                        src_node, source_interface.split(":")[0]
                    )
                    dst_runtime_intf = _runtime_interface_name(
                        dst_node, destination_interface.split(":")[0]
                    )
                    src_ip = src_node.cmd(
                        f"ip -4 -o addr show dev {src_runtime_intf} | awk '{{print $4}}'"
                    ).strip()
                    dst_ip = dst_node.cmd(
                        f"ip -4 -o addr show dev {dst_runtime_intf} | awk '{{print $4}}'"
                    ).strip()
                writer.writerow(
                    {
                        "EdgeId": edge_id,
                        "Source": link["src"],
                        "Destination": link["dst"],
                        "SourceInterface": source_interface,
                        "DestinationInterface": destination_interface,
                        "SourceIP": src_ip,
                        "DestinationIP": dst_ip,
                        "Profile": link["profile"],
                        "BandwidthMbps": link["attrs"]["bw"],
                        "Delay": link["attrs"]["delay"],
                        "LossPercent": link["attrs"]["loss"],
                    }
                )
    return node_path, link_path


def build_network(instance_id: str = ""):
    """Create the Mininet network from the topology specification, stopping it if construction fails."""
    from mininet.link import TCLink
    from mininet.net import Mininet

    spec = topology_spec()
    net = Mininet(link=TCLink, controller=None, autoSetMacs=True)

    try:
        net.instance_id = instance_id
        switches = {
            name: net.addHost(runtime_name(name, instance_id), ip=None) for name in spec["switches"]
        }
        hosts = {name: net.addHost(runtime_name(name, instance_id)) for name in spec["hosts"]}

        for logical, node in {**switches, **hosts}.items():
            setattr(node, LOGICAL_NAME_KEY, logical)

        for link in spec["links"]:
            src_name = str(link["src"])
            dst_name = str(link["dst"])
            src = switches.get(src_name) or hosts[src_name]
            dst = switches.get(dst_name) or hosts[dst_name]
            net.addLink(src, dst, **link["attrs"])
    except Exception as exc:
        print(f"warning: build_network failed; stopping partial Mininet state ({exc})", flush=True)
        try:
            net.stop()
        except Exception as stop_exc:
            print(f"warning: net.stop() during cleanup failed ({stop_exc})", flush=True)
        raise
    return net


def configure_network(net: object) -> dict[str, object]:
    """Assign addresses, start FRRouting with OSPF on every router, and return the convergence details."""
    from mininet.log import info

    info("*** Configuring L3 routed fabric\n")
    _configure_l3_addressing(net)
    if not _is_frr_available():
        raise RuntimeError("FRRouting is not available; Stage-1 generation requires OSPF routing")
    frr_ok, frr_details = _configure_frr_ospf(net)
    if not frr_ok:
        raise RuntimeError("FRR OSPF did not converge on every router")
    _wait_routing_settle()
    return dict(frr_details)


def _all_nodes(spec: dict[str, Any]) -> list[str]:
    """Return the router names followed by the host names of a specification."""
    return [*spec["switches"], *spec["hosts"]]


def _is_router(name: str) -> bool:
    """Return whether a logical node name belongs to a router."""
    return not name.startswith("h")


def _is_host(name: str) -> bool:
    """Return whether a logical node name belongs to a host."""
    return name.startswith("h")


def _wait_routing_settle() -> None:
    """Sleep briefly so programmed routes settle before traffic and telemetry start."""
    time.sleep(2)


def _link_interface_names(
    net: object, src_name: str, dst_name: str
) -> tuple[object, object, str, str]:
    """Return both endpoint nodes of a link with the runtime names of their connecting interfaces."""
    src_node = logical_node(net, src_name)
    dst_node = logical_node(net, dst_name)
    src_intf, dst_intf = src_node.connectionsTo(dst_node)[0]
    return src_node, dst_node, src_intf.name, dst_intf.name


def _link_ip_pair(link_index: int) -> tuple[str, str]:
    """Allocate /30 point-to-point subnets from 10.240.0.0/16."""
    base = link_index * 4
    octet2 = base // 256
    octet3 = base % 256
    return f"10.240.{octet2}.{octet3 + 1}/30", f"10.240.{octet2}.{octet3 + 2}/30"


def _host_ip(index: int) -> str:
    """Return the address of the host on the indexed access subnet."""
    return f"10.10.{index}.2/24"


def _host_gateway(index: int) -> str:
    """Return the router address that serves as gateway on the indexed access subnet."""
    return f"10.10.{index}.1"


def _configure_l3_addressing(net: object) -> None:
    """Address every link, give each host a default route, and enable forwarding on routers."""
    spec = topology_spec()
    host_counter = 1
    link_counter = 1

    for link in spec["links"]:
        src_name = str(link["src"])
        dst_name = str(link["dst"])
        src_node, dst_node, src_intf, dst_intf = _link_interface_names(net, src_name, dst_name)
        src_node.cmd(f"ip addr flush dev {src_intf}")
        dst_node.cmd(f"ip addr flush dev {dst_intf}")
        src_node.cmd(f"ip link set {src_intf} up")
        dst_node.cmd(f"ip link set {dst_intf} up")

        if _is_host(src_name) and _is_router(dst_name):
            host_ip = _host_ip(host_counter)
            gw_ip = _host_gateway(host_counter)
            src_node.cmd(f"ip addr add {host_ip} dev {src_intf}")
            dst_node.cmd(f"ip addr add {gw_ip}/24 dev {dst_intf}")
            src_node.cmd("ip route del default || true")
            src_node.cmd(f"ip route add default via {gw_ip} dev {src_intf}")
            host_counter += 1
            continue
        if _is_host(dst_name) and _is_router(src_name):
            host_ip = _host_ip(host_counter)
            gw_ip = _host_gateway(host_counter)
            dst_node.cmd(f"ip addr add {host_ip} dev {dst_intf}")
            src_node.cmd(f"ip addr add {gw_ip}/24 dev {src_intf}")
            dst_node.cmd("ip route del default || true")
            dst_node.cmd(f"ip route add default via {gw_ip} dev {dst_intf}")
            host_counter += 1
            continue

        src_ip, dst_ip = _link_ip_pair(link_counter)
        src_node.cmd(f"ip addr add {src_ip} dev {src_intf}")
        dst_node.cmd(f"ip addr add {dst_ip} dev {dst_intf}")
        link_counter += 1

    for router_name in spec["switches"]:
        router = logical_node(net, router_name)
        router.cmd("sysctl -w net.ipv4.ip_forward=1 >/dev/null 2>&1")


def _build_graph(spec: dict[str, Any]) -> dict[str, set[str]]:
    """Return the undirected adjacency sets of the topology keyed by node name."""
    graph: dict[str, set[str]] = {node: set() for node in _all_nodes(spec)}
    for link in spec["links"]:
        src = str(link["src"])
        dst = str(link["dst"])
        graph[src].add(dst)
        graph[dst].add(src)
    return graph


def node_primary_ip(node: object) -> str:
    """Return the first IPv4 address reported inside the node's namespace."""
    return node.cmd("hostname -I | awk '{print $1}'").strip()


def _host_prefixes(net: object, spec: dict[str, Any]) -> dict[str, str]:
    """Return the /24 access prefix of every host that reports an address."""
    prefixes: dict[str, str] = {}
    for host_name in spec["hosts"]:
        host = logical_node(net, host_name)
        ip = host.cmd("hostname -I | awk '{print $1}'").strip()
        if not ip:
            continue
        octets = ip.split(".")
        prefixes[host_name] = f"{octets[0]}.{octets[1]}.{octets[2]}.0/24"
    return prefixes


def _is_frr_available() -> bool:
    """Return whether the zebra, ospfd, and vtysh binaries are all installed."""
    binaries = _resolve_frr_binaries()
    return all(binaries.values())


def _resolve_frr_binaries() -> dict[str, str]:
    """Resolve FRR daemon binaries across common distro install layouts."""
    candidates: dict[str, tuple[str, ...]] = {
        "zebra": ("zebra", "/usr/lib/frr/zebra", "/usr/sbin/zebra"),
        "ospfd": ("ospfd", "/usr/lib/frr/ospfd", "/usr/sbin/ospfd"),
        "vtysh": ("vtysh", "/usr/bin/vtysh", "/usr/sbin/vtysh"),
    }
    resolved: dict[str, str] = {}
    for name, options in candidates.items():
        chosen = ""
        for option in options:
            if "/" in option:
                if Path(option).exists():
                    chosen = option
                    break
            else:
                path = shutil.which(option)
                if path:
                    chosen = path
                    break
        resolved[name] = chosen
    return resolved


def _router_interfaces(node: object) -> list[tuple[str, str]]:
    """Return the addressed interfaces of a router as name and CIDR pairs, excluding loopback."""
    interfaces: list[tuple[str, str]] = []
    for intf in node.intfList():
        name = str(intf.name)
        if name == "lo":
            continue
        cidr = node.cmd(f"ip -4 -o addr show dev {name} | awk '{{print $4}}'").strip()
        if cidr:
            interfaces.append((name, cidr))
    return interfaces


def _router_id_for_name(name: str) -> str:
    """Return a deterministic OSPF router identifier derived from the router's position in the specification."""
    switches = list(topology_spec()["switches"])
    if name not in switches:
        return "10.255.255.254"
    idx = switches.index(name) + 1
    octet3 = (idx - 1) // 254
    octet4 = ((idx - 1) % 254) + 1
    return f"10.255.{octet3}.{octet4}"


def _render_zebra_conf(logical_name_value: str) -> str:
    """Return the zebra daemon configuration text for one router."""
    return f"hostname {logical_name_value}\npassword zebra\nenable password zebra\nlog stdout\n"


def _render_ospfd_conf(logical_name_value: str, interfaces: list[tuple[str, str]]) -> str:
    """Return the ospfd configuration text that advertises every interface network in area 0."""
    router_id = _router_id_for_name(logical_name_value)
    lines = [
        f"hostname {logical_name_value}-ospfd",
        "password zebra",
        "enable password zebra",
        "log stdout",
        "router ospf",
        f" ospf router-id {router_id}",
    ]
    for _ifname, cidr in interfaces:
        network = str(ipaddress.ip_interface(cidr).network)
        lines.append(f" network {network} area 0.0.0.0")
    for ifname, cidr in interfaces:
        if ipaddress.ip_interface(cidr).network.prefixlen != 30:
            lines.append(f" passive-interface {ifname}")
    lines.append("exit")
    for ifname, cidr in interfaces:
        if ipaddress.ip_interface(cidr).network.prefixlen != 30:
            continue
        lines.extend(
            [
                f"interface {ifname}",
                " ip ospf network point-to-point",
                "exit",
            ]
        )
    return "\n".join(lines) + "\n"


def _stop_frr_daemons(net: object, runtime_dir: Path) -> int:
    """Stop every daemon from a partial or unconverged FRR startup."""
    stopped = 0
    for router_name in topology_spec()["switches"]:
        router = logical_node(net, router_name)
        router_dir = runtime_dir / str(getattr(router, "name"))
        for daemon in ("ospfd", "zebra"):
            pid_file = router_dir / f"{daemon}.pid"
            pid = router.cmd(f"cat {pid_file} 2>/dev/null || true").strip()
            if not pid:
                continue
            router.cmd(f"kill {pid} >/dev/null 2>&1 || true")
            stopped += 1
        router.cmd(
            f"rm -f {router_dir / 'ospfd.pid'} {router_dir / 'zebra.pid'} "
            f"{router_dir / 'ospfd.vty'} {router_dir / 'zebra.vty'} {router_dir / 'zebra.api'}"
        )
    return stopped


def _wait_for_frr_convergence(
    net: object,
    *,
    timeout_sec: float = 60.0,
    poll_interval_sec: float = 0.25,
) -> tuple[bool, dict[str, object]]:
    """Require every router to learn every non-direct host prefix via OSPF."""
    started_mono = time.monotonic()
    spec = topology_spec()
    graph = _build_graph(spec)
    host_prefixes = _host_prefixes(net, spec)
    if len(host_prefixes) != len(spec["hosts"]):
        return False, {
            "frr_converged_router_count": 0,
            "frr_convergence_missing_routes": "host_prefix_discovery_incomplete",
        }

    expected_by_router: dict[str, set[str]] = {}
    for router_name in spec["switches"]:
        expected_by_router[router_name] = {
            prefix
            for host_name, prefix in host_prefixes.items()
            if host_name not in graph[router_name]
        }

    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    missing_by_router: dict[str, list[str]] = {}
    while True:
        missing_by_router = {}
        for router_name, expected in expected_by_router.items():
            router = logical_node(net, router_name)
            output = router.cmd("ip -4 route show proto ospf")
            learned = {
                line.split()[0]
                for line in output.splitlines()
                if line.strip() and line.split()[0] != "default"
            }
            missing = sorted(expected - learned)
            if missing:
                missing_by_router[router_name] = missing
        if not missing_by_router:
            break
        now = time.monotonic()
        if now >= deadline:
            break
        time.sleep(min(max(0.001, poll_interval_sec), deadline - now))

    max_error_chars = 4000
    missing_text = " | ".join(
        f"{router}:{','.join(prefixes)}" for router, prefixes in sorted(missing_by_router.items())
    )[:max_error_chars]
    return not missing_by_router, {
        "frr_converged_router_count": len(spec["switches"]) - len(missing_by_router),
        "frr_convergence_missing_routes": missing_text,
        "frr_convergence_wait_sec": round(time.monotonic() - started_mono, 6),
    }


def _all_frr_routers_started(started: int, expected: int) -> bool:
    """FRR mode requires every expected router to have started."""
    return expected > 0 and started == expected


def _capture_frr_failure_diagnostics(net: object, runtime_dir: Path) -> dict[str, str]:
    """Capture bounded OSPF state before partial FRR cleanup removes its sockets."""
    vtysh_bin = _resolve_frr_binaries().get("vtysh", "")
    if not vtysh_bin:
        return {"frr_ospf_diagnostics": "vtysh_unavailable"}
    entries: list[str] = []
    for router_name in topology_spec()["switches"]:
        router = logical_node(net, router_name)
        router_dir = runtime_dir / str(getattr(router, "name"))
        command = (
            f"{shlex.quote(vtysh_bin)} --vty_socket {shlex.quote(str(router_dir))} "
            f"-d ospfd -c {shlex.quote('show ip ospf neighbor')} "
            f"-c {shlex.quote('show ip ospf route')} 2>&1"
        )
        output = router.cmd(command).strip()
        entries.append(f"{router_name}:{output or 'no_output'}")
    return {"frr_ospf_diagnostics": " | ".join(entries)[:8000]}


def _configure_frr_ospf(net: object) -> tuple[bool, dict[str, object]]:
    """Start zebra and ospfd on every router, wait for OSPF convergence, and report the outcome."""
    spec = topology_spec()
    instance_id = str(getattr(net, "instance_id", "") or "default")
    runtime_dir = Path("/tmp/ridge_frr") / f"{instance_id}-p{os.getpid()}"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    # FRR daemons can drop privileges to the packaged "frr" user.
    # Ensure writable runtime dirs for pid/socket creation.
    os.chmod(runtime_dir, 0o777)
    try:
        frr_uid = pwd.getpwnam("frr").pw_uid
        frr_gid = pwd.getpwnam("frr").pw_gid
    except KeyError:
        frr_uid = -1
        frr_gid = -1
    binaries = _resolve_frr_binaries()
    zebra_bin = binaries.get("zebra", "")
    ospfd_bin = binaries.get("ospfd", "")
    if not zebra_bin or not ospfd_bin:
        return False, {
            "frr_router_count": len(spec["switches"]),
            "frr_start_success_count": 0,
            "frr_start_errors": "missing_zebra_or_ospfd_binary",
        }

    started = 0
    startup_errors: list[str] = []
    for router_name in spec["switches"]:
        router = logical_node(net, router_name)
        logical_router = logical_name(router)
        runtime_router = str(getattr(router, "name"))
        interfaces = _router_interfaces(router)
        if not interfaces:
            continue

        router_dir = runtime_dir / runtime_router
        router_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(router_dir, 0o777)
        if frr_uid >= 0 and frr_gid >= 0:
            try:
                os.chown(router_dir, frr_uid, frr_gid)
            except PermissionError:
                pass
        zebra_conf = router_dir / "zebra.conf"
        ospfd_conf = router_dir / "ospfd.conf"
        zebra_pid = router_dir / "zebra.pid"
        ospfd_pid = router_dir / "ospfd.pid"
        zebra_api = router_dir / "zebra.api"

        zebra_conf.write_text(_render_zebra_conf(logical_router), encoding="utf-8")
        ospfd_conf.write_text(_render_ospfd_conf(logical_router, interfaces), encoding="utf-8")
        if frr_uid >= 0 and frr_gid >= 0:
            try:
                os.chown(zebra_conf, frr_uid, frr_gid)
                os.chown(ospfd_conf, frr_uid, frr_gid)
            except PermissionError:
                pass

        # Ensure stale daemon processes are removed before restart.
        old_zebra_pid = router.cmd(f"cat {zebra_pid} 2>/dev/null || true").strip()
        old_ospfd_pid = router.cmd(f"cat {ospfd_pid} 2>/dev/null || true").strip()
        if old_zebra_pid:
            router.cmd(f"kill {old_zebra_pid} >/dev/null 2>&1 || true")
        if old_ospfd_pid:
            router.cmd(f"kill {old_ospfd_pid} >/dev/null 2>&1 || true")
        for stale in (
            zebra_pid,
            ospfd_pid,
            zebra_api,
            router_dir / "zebra.vty",
            router_dir / "ospfd.vty",
        ):
            router.cmd(f"rm -f {stale}")

        zebra_cmd = (
            f"{zebra_bin} -d "
            f"-f {zebra_conf} "
            f"-i {zebra_pid} "
            f"-z {zebra_api} "
            f"-A 127.0.0.1 "
            f"--vty_socket {router_dir}"
        )
        ospfd_cmd = (
            f"{ospfd_bin} -d "
            f"-f {ospfd_conf} "
            f"-i {ospfd_pid} "
            f"-z {zebra_api} "
            f"-A 127.0.0.1 "
            f"--vty_socket {router_dir}"
        )
        zebra_out = router.cmd(zebra_cmd).strip()
        ospfd_out = router.cmd(ospfd_cmd).strip()

        if zebra_out:
            startup_errors.append(f"{logical_router}:zebra:{zebra_out}")
        if ospfd_out:
            startup_errors.append(f"{logical_router}:ospfd:{ospfd_out}")

        zebra_alive = False
        ospfd_alive = False
        zebra_socket_ready = False
        vty_ready = False
        for _ in range(20):
            zebra_pid_value = router.cmd(f"cat {zebra_pid} 2>/dev/null").strip()
            ospfd_pid_value = router.cmd(f"cat {ospfd_pid} 2>/dev/null").strip()
            if zebra_pid_value:
                zebra_alive = (
                    router.cmd(f"kill -0 {zebra_pid_value} >/dev/null 2>&1; echo $?").strip() == "0"
                )
            if ospfd_pid_value:
                ospfd_alive = (
                    router.cmd(f"kill -0 {ospfd_pid_value} >/dev/null 2>&1; echo $?").strip() == "0"
                )
            zebra_socket_ready = router.cmd(f"test -S {zebra_api}; echo $?").strip() == "0"
            vty_ready = (
                router.cmd(f"test -S {router_dir / 'zebra.vty'}; echo $?").strip() == "0"
                and router.cmd(f"test -S {router_dir / 'ospfd.vty'}; echo $?").strip() == "0"
            )
            if zebra_alive and ospfd_alive and zebra_socket_ready and vty_ready:
                started += 1
                break
            time.sleep(0.25)

        if not (zebra_alive and ospfd_alive and zebra_socket_ready and vty_ready):
            startup_errors.append(
                f"{logical_router}:health zebra_alive={zebra_alive} ospfd_alive={ospfd_alive} "
                f"zebra_api={zebra_socket_ready} vty={vty_ready}"
            )

    router_count = len(spec["switches"])
    all_started = _all_frr_routers_started(started, router_count)
    convergence_ok = False
    convergence_details: dict[str, object] = {
        "frr_converged_router_count": 0,
        "frr_convergence_missing_routes": "startup_incomplete",
    }
    if all_started:
        convergence_ok, convergence_details = _wait_for_frr_convergence(net)

    failure_diagnostics: dict[str, str] = {}
    cleanup_count = 0
    if not all_started or not convergence_ok:
        if all_started:
            failure_diagnostics = _capture_frr_failure_diagnostics(net, runtime_dir)
        cleanup_count = _stop_frr_daemons(net, runtime_dir)

    max_error_chars = 4000
    return all_started and convergence_ok, {
        "frr_router_count": len(spec["switches"]),
        "frr_start_success_count": started,
        "frr_start_errors": " | ".join(startup_errors)[:max_error_chars],
        "frr_runtime_dir": str(runtime_dir),
        "frr_partial_cleanup_process_count": cleanup_count,
        **failure_diagnostics,
        **convergence_details,
    }
