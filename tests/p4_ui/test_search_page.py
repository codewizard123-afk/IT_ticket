"""P4-T4: Search & filter page (FR-07)."""

from __future__ import annotations

from tests.p4_ui.helpers import all_text, click, page


def search(make_client, role: str = "admin", **fields):
    at = page("3_Search_and_filter.py", make_client(role))
    for key, value in fields.items():
        widget = at.selectbox(key=key) if key != "s_q" else at.text_input(key=key)
        if key == "s_q":
            widget.input(value)
        else:
            widget.select(value)
    at = click(at, "Search")
    assert not at.exception, at.exception
    return at


def ids(at) -> set[str]:
    return set(at.dataframe[0].value["Ticket"]) if at.dataframe else set()


def test_default_lists_tickets_with_synthetic_flags(make_client):
    at = page("3_Search_and_filter.py", make_client("admin"))
    assert not at.exception
    df = at.dataframe[0].value
    assert len(df) == 20
    assert "SYNTHETIC" in set(df["Resolution origin"])
    assert "VERIFIED" in set(df["Resolution origin"])


def test_filters_combine(make_client):
    net = ids(search(make_client, s_category="Network", s_size=50))
    vpn_at = search(make_client, s_category="Network", s_subcategory="VPN", s_size=50)
    vpn = ids(vpn_at)
    assert vpn and vpn < net
    assert set(vpn_at.dataframe[0].value["Subcategory"]) == {"VPN"}
    clash = search(make_client, s_category="Network", s_subcategory="Oracle")
    assert not clash.dataframe and any("No tickets" in str(i.value) for i in clash.info)


def test_priority_and_group_filters(make_client):
    at = search(make_client, s_priority="Low", s_group="SERVICE-DESK", s_size=50)
    df = at.dataframe[0].value
    assert len(df) > 0
    assert set(df["Priority"]) == {"Low"} and set(df["Group"]) == {"SERVICE-DESK"}


def test_text_query(make_client):
    at = search(make_client, s_q="gateway not responding")
    assert "TKT-0001" in ids(at)


def test_pagination(make_client):
    at = page("3_Search_and_filter.py", make_client("admin"))
    first = ids(at)
    at = click(at, "Next ▶")
    second = ids(at)
    assert first and second and first.isdisjoint(second)
    assert "page 2 of" in all_text(at)
    at = click(at, "◀ Previous")
    assert ids(at) == first


def test_restricted_rows_hidden_for_l1(make_client, backend):
    rid = next(t for t, r in backend.ticket_store._recs.items() if r.restricted)
    admin = search(make_client, "admin", s_size=50)
    l1 = search(make_client, "l1", s_size=50)
    assert rid in ids(admin)
    assert rid not in ids(l1)
    assert len(ids(admin)) == len(ids(l1)) + 1  # everything else is still visible


def test_detail_drawer(make_client):
    at = search(make_client, s_q="gateway not responding")
    at.selectbox(key="s_open").select("TKT-0001").run()
    assert not at.exception
    assert any("TKT-0001" in e.label for e in at.expander)
    assert "Resolution (synthetic)" in all_text(at)
