"""Find a step-free route from the street to a platform and describe it."""

import heapq
from dataclasses import dataclass, field

# A lift costs far more than a walkway, so the search uses as few lifts as
# possible first and as few links as possible second.
LIFT_COST = 1000
LINK_COST = 1

OK = "ok"            # a route exists right now
BLOCKED = "blocked"  # routes exist, but broken lifts block all of them
NO_ROUTE = "none"    # TfL's data has no step-free route at all
NO_INFO = "unknown"  # no route, and TfL hasn't surveyed this platform


@dataclass
class RouteResult:
    status: str
    places: list = field(default_factory=list)   # place ids, street first
    links: list = field(default_factory=list)    # link dicts, in order
    steps: list = field(default_factory=list)    # readable directions
    blocking_lifts: list = field(default_factory=list)  # lift names
    # True when the route starts at an entrance TfL's data doesn't link to
    # the street (only used when no route from the street exists).
    unlinked_entrance: bool = False


def _search(routes, starts, platform_id, broken):
    """Dijkstra from ``starts``. Returns (places, links) or None.

    The search state is (place, lift we arrived in), so staying in the same
    lift through an intermediate stop costs one lift, not two. Ties are
    broken by place id, then by lift name, so results are stable.
    """
    adjacency = {}
    for link in routes["links"]:
        if link["kind"] == "lift" and link["lift_id"] in broken:
            continue
        adjacency.setdefault(link["a"], []).append((link["b"], link))
        adjacency.setdefault(link["b"], []).append((link["a"], link))
    for edges in adjacency.values():
        edges.sort(key=lambda edge: (edge[0], edge[1].get("lift_name") or ""))

    best = {}
    previous = {}
    heap = []
    for start in starts:
        state = (start, "")
        best[state] = 0
        heapq.heappush(heap, (0, state))
    end_state = None
    while heap:
        cost, state = heapq.heappop(heap)
        if cost > best.get(state, float("inf")):
            continue
        node, riding = state
        if node == platform_id:
            end_state = state
            break
        for other, link in adjacency.get(node, []):
            if link["kind"] == "lift":
                lift_id = link["lift_id"]
                step = LINK_COST if lift_id == riding else LIFT_COST
            else:
                lift_id = ""
                step = LINK_COST
            next_state = (other, lift_id)
            if cost + step < best.get(next_state, float("inf")):
                best[next_state] = cost + step
                previous[next_state] = (state, link)
                heapq.heappush(heap, (cost + step, next_state))

    if end_state is None:
        return None
    states, links = [end_state], []
    while states[-1] in previous:
        state, link = previous[states[-1]]
        links.append(link)
        states.append(state)
    states.reverse()
    links.reverse()
    return [node for node, _ in states], links


def _search_from_street_first(routes, platform_id, broken):
    """(places, links, unlinked) or None.

    ``starts`` is the street node, then any street-level entrances TfL's
    data doesn't link to the street. Those may have steps in reality (the
    file holds step-free links only), so they are used only when the street
    itself has no route — never because they are a link shorter.
    """
    street = [start for start in routes["starts"] if routes["places"][start]["kind"] == "street"]
    found = _search(routes, street, platform_id, broken)
    if found:
        return (*found, False)
    found = _search(routes, routes["starts"], platform_id, broken)
    return (*found, True) if found else None


def find_route(routes, platform_id, broken_lift_ids=()):
    """Route from the street to ``platform_id``, avoiding broken lifts."""
    broken = set(broken_lift_ids)
    found = _search_from_street_first(routes, platform_id, broken)
    if found:
        places, links, unlinked = found
        return RouteResult(OK, places, links, describe(routes, places, links),
                           unlinked_entrance=unlinked)
    unblocked = _search_from_street_first(routes, platform_id, set())
    if not unblocked:
        surveyed = routes["platforms"][platform_id].get("route_info", True)
        return RouteResult(NO_ROUTE if surveyed else NO_INFO)
    names = []
    for link in unblocked[1]:
        if link["kind"] == "lift" and link["lift_id"] in broken and link["lift_name"] not in names:
            names.append(link["lift_name"])
    if not names:
        # The fault-free route avoids the broken lift, yet every route that
        # avoids it fails, so name every broken lift at the station.
        for link in routes["links"]:
            if link["kind"] == "lift" and link["lift_id"] in broken and link["lift_name"] not in names:
                names.append(link["lift_name"])
    return RouteResult(BLOCKED, blocking_lifts=names)


def describe(routes, places, links):
    """Turn a route into numbered-step text (without the numbers)."""
    info = routes["places"]
    platform = routes["platforms"][places[-1]]
    steps = []
    index = 0
    if info[places[0]]["kind"] == "street":
        if not links:
            return steps
        if links[0]["kind"] == "walkway":
            steps.append(f"Enter at {info[places[1]]['name']}")
            index = 1
        else:
            # A lift or ramp straight from the street: describe it as a step
            # of its own rather than folding it into "Enter at".
            steps.append("Start at street level")
    else:
        steps.append(f"Enter at {info[places[0]]['name']}")

    i = index
    while i < len(links):
        link = links[i]
        if link["kind"] == "lift":
            # A lift with intermediate stops is several links: ride it once.
            j = i
            while j + 1 < len(links) and links[j + 1]["kind"] == "lift" \
                    and links[j + 1]["lift_id"] == link["lift_id"]:
                j += 1
            start_level = info[places[i]]["level"]
            end = places[j + 1]
            end_level = info[end]["level"]
            direction = " up" if end_level > start_level else " down" if end_level < start_level else ""
            small = " (small lift)" if link["small"] else ""
            steps.append(f"Take {link['lift_name']}{small}{direction} to {info[end]['name']}")
            i = j + 1
        elif link["kind"] == "ramp":
            steps.append(f"Use the ramp to {info[places[i + 1]]['name']}")
            i += 1
        else:
            # Run of walkways: one step to where the run ends.
            j = i
            while j + 1 < len(links) and links[j + 1]["kind"] == "walkway":
                j += 1
            steps.append(f"Go to {info[places[j + 1]]['name']}")
            i = j + 1

    details = []
    if platform["step"]:
        details.append(f"step to train {platform['step']}")
    if platform["gap"]:
        details.append(f"gap {platform['gap']}")
    if platform["manual_ramp"]:
        details.append("staff ramp available")
    if details:
        steps.append(f"{platform['name']}: {', '.join(details)}")
    return steps
