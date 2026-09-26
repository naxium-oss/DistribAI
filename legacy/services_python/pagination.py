"""
Pagination Utilities for API Responses
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PaginationParams:
    page: int = 1
    per_page: int = 20
    cursor: str | None = None
    sort_by: str = "created_at"
    sort_order: str = "desc"

    def __post_init__(self):
        self.page = max(1, self.page)
        self.per_page = max(1, min(100, self.per_page))
        self.sort_order = (
            "desc" if self.sort_order.lower() not in ("asc", "desc") else self.sort_order.lower()
        )


@dataclass
class PaginatedResponse:
    data: list[Any]
    pagination: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"data": self.data, "pagination": self.pagination}


def parse_pagination_params(
    query_params: dict[str, Any],
    default_per_page: int = 20,
    max_per_page: int = 100,
    allowed_sort_columns: set[str] | None = None,
) -> PaginationParams:
    page = 1
    if "page" in query_params:
        try:
            page = int(query_params["page"])
        except (ValueError, TypeError):
            pass
    per_page = default_per_page
    if "per_page" in query_params:
        try:
            per_page = int(query_params["per_page"])
        except (ValueError, TypeError):
            pass
    per_page = max(1, min(max_per_page, per_page))
    cursor = query_params.get("cursor")
    sort_by = query_params.get("sort_by", "created_at")
    if allowed_sort_columns and sort_by not in allowed_sort_columns:
        sort_by = "created_at"
    if not all(c.isalnum() or c == "_" for c in sort_by):
        sort_by = "created_at"
    sort_order = query_params.get("sort_order", "desc").lower()
    if sort_order not in ("asc", "desc"):
        sort_order = "desc"
    return PaginationParams(
        page=page, per_page=per_page, cursor=cursor, sort_by=sort_by, sort_order=sort_order
    )


def paginate_list(items: list[Any], params: PaginationParams) -> PaginatedResponse:
    total = len(items)
    start_idx = (params.page - 1) * params.per_page
    end_idx = start_idx + params.per_page
    page_data = items[start_idx:end_idx]
    total_pages = (total + params.per_page - 1) // params.per_page
    pagination = {
        "page": params.page,
        "per_page": params.per_page,
        "total": total,
        "total_pages": total_pages,
        "has_next": params.page < total_pages,
        "has_prev": params.page > 1,
        "sort_by": params.sort_by,
        "sort_order": params.sort_order,
    }
    if pagination["has_next"]:
        pagination["next_page"] = params.page + 1
    if pagination["has_prev"]:
        pagination["prev_page"] = params.page - 1
    return PaginatedResponse(data=page_data, pagination=pagination)


def paginate_database_query(
    conn,
    base_query: str,
    count_query: str,
    params: PaginationParams,
    query_args: tuple = (),
    allowed_sort_columns: set[str] | None = None,
) -> PaginatedResponse:
    total_row = conn.execute(count_query, query_args).fetchone()
    total = total_row[0] if total_row else 0
    sort_by = params.sort_by
    if allowed_sort_columns and sort_by not in allowed_sort_columns:
        sort_by = "created_at"
    sort_order = params.sort_order.upper()
    if sort_order not in ("ASC", "DESC"):
        sort_order = "DESC"
    paginated_query = f"""
        {base_query}
        ORDER BY {sort_by} {sort_order}
        LIMIT ? OFFSET ?
    """
    offset = (params.page - 1) * params.per_page
    query_args_with_pagination = query_args + (params.per_page, offset)
    cursor = conn.execute(paginated_query, query_args_with_pagination)
    rows = cursor.fetchall()
    data = [dict(row) for row in rows]
    total_pages = (total + params.per_page - 1) // params.per_page if total > 0 else 1
    pagination = {
        "page": params.page,
        "per_page": params.per_page,
        "total": total,
        "total_pages": total_pages,
        "has_next": params.page < total_pages,
        "has_prev": params.page > 1,
        "sort_by": params.sort_by,
        "sort_order": params.sort_order,
    }
    if pagination["has_next"]:
        pagination["next_page"] = params.page + 1
    if pagination["has_prev"]:
        pagination["prev_page"] = params.page - 1
    return PaginatedResponse(data=data, pagination=pagination)


def build_pagination_links(base_url: str, params: PaginationParams, total_pages: int) -> dict:
    links = {}
    links["self"] = f"{base_url}?page={params.page}&per_page={params.per_page}"
    links["first"] = f"{base_url}?page=1&per_page={params.per_page}"
    links["last"] = f"{base_url}?page={total_pages}&per_page={params.per_page}"
    if params.page < total_pages:
        links["next"] = f"{base_url}?page={params.page + 1}&per_page={params.per_page}"
    if params.page > 1:
        links["prev"] = f"{base_url}?page={params.page - 1}&per_page={params.per_page}"
    return links


class PaginationHeaders:
    @staticmethod
    def build(total: int, page: int, per_page: int) -> dict:
        total_pages = (total + per_page - 1) // per_page if total > 0 else 1
        return {
            "X-Total-Count": str(total),
            "X-Page": str(page),
            "X-Per-Page": str(per_page),
            "X-Total-Pages": str(total_pages),
        }
