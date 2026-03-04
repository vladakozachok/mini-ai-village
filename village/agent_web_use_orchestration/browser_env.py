"""Playwright browser session class."""

from __future__ import annotations

import base64
import io
from dataclasses import asdict, dataclass
from typing import Any

from PIL import Image
from playwright.async_api import Browser, BrowserContext, Page, async_playwright

import village.config as cfg
from village.agent_web_use_orchestration.actions import Action, ActionType
from village.agent_web_use_orchestration.observation import (
    OBSERVATION_SCRIPT,
    build_observation_from_raw,
)


@dataclass
class ExecutionResult:
    success: bool
    error_message: str | None = None
    data: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class BrowserEnvironment:
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        headless: bool = True,
    ) -> None:
        self.width = width
        self.height = height
        self.headless = headless

        self._playwright = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None

    async def launch(self) -> None:
        self._playwright = await async_playwright().start()
        self.browser = await self._playwright.chromium.launch(headless=self.headless)
        self.context = await self.browser.new_context(
            viewport={"width": self.width, "height": self.height}
        )
        self.page = await self.context.new_page()

    async def close(self) -> None:
        for obj, method in (
            (self.page, "close"),
            (self.context, "close"),
            (self.browser, "close"),
            (self._playwright, "stop"),
        ):
            if obj is not None:
                try:
                    await getattr(obj, method)()
                except Exception:
                    pass

    async def goto(self, url: str) -> ExecutionResult:
        if not self.page:
            return ExecutionResult(
                success=False, error_message="Browser not initialised."
            )
        try:
            await self.page.goto(url, wait_until="domcontentloaded")
            return ExecutionResult(success=True)
        except Exception as exc:
            data: dict[str, Any] = {"url": self.page.url}
            try:
                data["title"] = await self.page.title()
            except Exception:
                pass
            return ExecutionResult(success=False, error_message=str(exc), data=data)

    async def get_current_url(self) -> str:
        if not self.page:
            raise RuntimeError("Browser not initialised.")
        return self.page.url

    async def build_observation(self) -> dict[str, Any]:
        if not self.page:
            raise RuntimeError("Browser not initialised.")

        if cfg.PAGE_SETTLE_MS > 0:
            await self.page.wait_for_timeout(cfg.PAGE_SETTLE_MS)

        raw = await self.page.evaluate(
            OBSERVATION_SCRIPT,
            {
                "detail": "compact",
                "maxElements": 100,
                "maxTargetSurfaces": cfg.MAX_OBSERVATION_TARGET_SURFACES,
                "maxSurfaceAnchors": cfg.MAX_OBSERVATION_SURFACE_ANCHORS,
                "maxPromotedAnchors": cfg.MAX_OBSERVATION_PROMOTED_ANCHORS,
            },
        )
        return build_observation_from_raw(raw=raw, url=self.page.url, detail="compact")

    async def execute(self, action: Action) -> ExecutionResult:
        if not self.page:
            return ExecutionResult(
                success=False, error_message="Browser not initialised."
            )
        try:
            return await self._dispatch(action)
        except Exception as exc:
            data: dict[str, Any] = {"url": self.page.url}
            try:
                data["title"] = await self.page.title()
            except Exception:
                pass
            return ExecutionResult(success=False, error_message=str(exc), data=data)

    async def _dispatch(self, action: Action) -> ExecutionResult:
        match action.type:

            case ActionType.NAVIGATE:
                await self.page.goto(action.url, wait_until="domcontentloaded")
                return ExecutionResult(success=True)

            case ActionType.SCREENSHOT:
                return await self._take_screenshot()

            case ActionType.CLICK:
                return await self._click(action)

            case ActionType.CLICK_RELATIVE:
                return await self._click_relative(action)

            case ActionType.CLICK_INDEX:
                raise ValueError(
                    "click_index must be resolved to click before execution."
                )

            case ActionType.SCROLL:
                dx = (
                    action.amount
                    if action.direction == "right"
                    else (-action.amount if action.direction == "left" else 0)
                )
                dy = (
                    action.amount
                    if action.direction == "down"
                    else (-action.amount if action.direction == "up" else 0)
                )
                await self.page.mouse.wheel(dx, dy)
                return ExecutionResult(success=True)

            case ActionType.KEYPRESS:
                for key in action.keys or []:
                    await self.page.keyboard.press(key)
                return ExecutionResult(success=True)

            case ActionType.TYPE:
                await self.page.fill(action.selector, action.text)
                return ExecutionResult(success=True)

            case ActionType.GET_VALUE:
                value = await self._get_value(action.selector)
                return ExecutionResult(success=True, data={"value": value})

            case _:
                raise ValueError(f"Unsupported action type: {action.type}")

    async def screenshot(self) -> bytes:
        if not self.page:
            raise RuntimeError("Browser not initialised.")
        return await self.page.screenshot(full_page=False)

    async def _take_screenshot(self) -> ExecutionResult:
        png = await self.page.screenshot(full_page=False)
        img = Image.open(io.BytesIO(png))
        small = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
        buf = io.BytesIO()
        small.save(buf, format="JPEG", quality=50, optimize=True)
        return ExecutionResult(
            success=True,
            data={
                "screenshot_base64": base64.b64encode(buf.getvalue()).decode(),
                "media_type": "image/jpeg",
                "viewport_width": self.width,
                "viewport_height": self.height,
                "note": (
                    "Image is at half resolution for token efficiency; "
                    "all coordinates refer to the ORIGINAL viewport."
                ),
            },
        )

    async def _click(self, action: Action) -> ExecutionResult:
        pre = await self._page_state()

        if action.selector:
            await self.page.click(action.selector)
            coords = None
        elif action.x is not None and action.y is not None:
            coords = (float(action.x), float(action.y))
            await self.page.mouse.click(*coords)
        else:
            raise ValueError("click requires selector or x/y coordinates.")

        await self.page.wait_for_timeout(cfg.ACTION_SETTLE_MS)
        post = await self._page_state()

        return ExecutionResult(
            success=True,
            data={
                "selector": action.selector,
                "resolved_x": round(coords[0], 2) if coords else None,
                "resolved_y": round(coords[1], 2) if coords else None,
                "state_changed": self._state_changed(pre, post),
            },
        )

    async def _click_relative(self, action: Action) -> ExecutionResult:
        element = await self.page.query_selector(action.selector)
        if not element:
            raise ValueError(f"Selector not found: {action.selector!r}")
        bbox = await element.bounding_box()
        if not bbox:
            raise ValueError(f"Element has no bounding box: {action.selector!r}")

        rel_x = max(0.0, min(1.0, float(action.rel_x)))
        rel_y = max(0.0, min(1.0, float(action.rel_y)))
        x = bbox["x"] + bbox["width"] * rel_x
        y = bbox["y"] + bbox["height"] * rel_y

        pre = await self._page_state()
        await self.page.mouse.click(x, y)
        await self.page.wait_for_timeout(cfg.ACTION_SETTLE_MS)
        post = await self._page_state()

        return ExecutionResult(
            success=True,
            data={
                "selector": action.selector,
                "rel_x": rel_x,
                "rel_y": rel_y,
                "resolved_x": round(x, 2),
                "resolved_y": round(y, 2),
                "target_bbox": {k: round(v, 2) for k, v in bbox.items()},
                "state_changed": self._state_changed(pre, post),
            },
        )

    async def _get_value(self, selector: str) -> str:
        element = await self.page.query_selector(selector)
        if not element:
            raise ValueError(f"Selector not found: {selector!r}")
        tag = (await element.evaluate("el => el.tagName")).lower()
        if tag in {"input", "textarea", "select"}:
            return await element.evaluate("el => el.value")
        return await element.evaluate("el => el.textContent")

    async def _page_state(self) -> dict[str, Any]:
        if not self.page:
            return {}
        try:
            return await self.page.evaluate("""() => ({
                url:               window.location.href,
                title:             document.title || '',
                scroll_x:          Math.round(window.scrollX || 0),
                scroll_y:          Math.round(window.scrollY || 0),
                body_text_excerpt: (document.body?.innerText || '')
                                     .replace(/\\s+/g, ' ').trim().slice(0, 240),
                board_fingerprint: (() => {
                    const board = document.querySelector('cg-board');
                    if (!board) return '';
                    return Array.from(
                        board.querySelectorAll(
                            'piece, square.selected, square.move-dest, square.last-move, square.check'
                        )
                    )
                        .slice(0, 96)
                        .map((el) => {
                            const tag = (el.tagName || '').toLowerCase();
                            const cls = String(el.className || '');
                            const cgKey = el.getAttribute('cgkey') || '';
                            const style = el.getAttribute('style') || '';
                            return `${tag}:${cls}:${cgKey}:${style}`;
                        })
                        .join('|')
                        .slice(0, 2000);
                })(),
            })""")
        except Exception:
            return {}

    @staticmethod
    def _state_changed(pre: dict, post: dict) -> bool:
        return any(
            pre.get(k) != post.get(k)
            for k in (
                "url",
                "title",
                "scroll_x",
                "scroll_y",
                "body_text_excerpt",
                "board_fingerprint",
            )
        )
