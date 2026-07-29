"""
ARIA Menubar App
Runs ARIA as a Mac menubar app with a global Option+Space hotkey
that summons/dismisses a floating WKWebView window.
"""

import sys
import threading
import time
import subprocess
import urllib.request
import urllib.error
import webbrowser
from pathlib import Path

# Ensure the aria root is on the path
ARIA_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ARIA_ROOT))

import objc
import rumps

from AppKit import (
    NSApplication,
    NSApp,
    NSWindow,
    NSWindowStyleMaskTitled,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskMiniaturizable,
    NSWindowStyleMaskResizable,
    NSWindowStyleMaskFullSizeContentView,
    NSBackingStoreBuffered,
    NSMakeRect,
    NSScreen,
    NSColor,
    NSObject,
    NSRunLoop,
    NSDefaultRunLoopMode,
    NSDate,
)
from WebKit import WKWebView, WKWebViewConfiguration
import Quartz
from Quartz import (
    CGEventTapCreate,
    CGEventTapEnable,
    CFMachPortCreateRunLoopSource,
    CFRunLoopAddSource,
    CFRunLoopGetCurrent,
    kCGSessionEventTap,
    kCGHeadInsertEventTap,
    kCGEventKeyDown,
    CGEventGetFlags,
    CGEventGetIntegerValueField,
    kCGKeyboardEventKeycode,
    kCGEventFlagMaskAlternate,
)
import CoreFoundation


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_W = 480
WINDOW_H = 700
GRADIO_URL = "http://localhost:7860"
OPTION_SPACE_KEYCODE = 49  # Space bar keycode


# ---------------------------------------------------------------------------
# Backend launcher
# ---------------------------------------------------------------------------

def _ensure_dependencies():
    """Install requirements with python3.11 if langgraph is not importable."""
    try:
        import importlib
        importlib.import_module("langgraph")
    except ImportError:
        import subprocess
        req_path = str(ARIA_ROOT / "requirements.txt")
        print("[ARIA menubar] langgraph not found — installing requirements…", file=sys.stderr)
        subprocess.run(
            ["pip3.11", "install", "-r", req_path],
            check=True,
        )


def _start_aria_backend():
    """Import and run main.main() in this process on a daemon thread."""
    try:
        _ensure_dependencies()
        import main as aria_main
        aria_main.main()
    except SystemExit:
        pass
    except Exception as e:
        print(f"[ARIA menubar] Backend error: {e}", file=sys.stderr)


def _poll_until_ready(timeout: int = 60) -> bool:
    """Return True when Gradio responds, False on timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(GRADIO_URL, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


# ---------------------------------------------------------------------------
# Floating WKWebView window
# ---------------------------------------------------------------------------

class ARIAWindowDelegate(NSObject):
    """Delegate: hide on Escape, hide on window-lost-focus."""

    def init(self):
        self = objc.super(ARIAWindowDelegate, self).init()
        self._controller = None
        return self

    def setController_(self, controller):
        self._controller = controller

    # Called when window resigns key (clicked outside)
    def windowDidResignKey_(self, notification):
        if self._controller:
            self._controller.hide_window()

    # Called when window is about to close — intercept and hide instead
    def windowShouldClose_(self, sender):
        if self._controller:
            self._controller.hide_window()
        return False


class ARIAWindowController:
    """Owns the NSWindow + WKWebView."""

    def __init__(self):
        self._window = None
        self._webview = None
        self._delegate = None
        self._visible = False

    def _build(self):
        """Lazily create the window (must run on main thread)."""
        if self._window is not None:
            return

        # Position: bottom-right corner of main screen
        screen = NSScreen.mainScreen()
        sf = screen.visibleFrame()
        x = sf.origin.x + sf.size.width - WINDOW_W - 20
        y = sf.origin.y + 20
        rect = NSMakeRect(x, y, WINDOW_W, WINDOW_H)

        style = (
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskMiniaturizable
            | NSWindowStyleMaskResizable
            | NSWindowStyleMaskFullSizeContentView
        )

        self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, style, NSBackingStoreBuffered, False
        )
        self._window.setTitle_("ARIA")
        self._window.setBackgroundColor_(NSColor.blackColor())

        # Floating level — stays above normal windows
        self._window.setLevel_(Quartz.kCGFloatingWindowLevel)

        # Titlebar transparent / merged with content
        self._window.setTitlebarAppearsTransparent_(True)
        self._window.setMovableByWindowBackground_(True)

        # WKWebView filling the window
        cfg = WKWebViewConfiguration.alloc().init()
        content_rect = NSMakeRect(0, 0, WINDOW_W, WINDOW_H)
        self._webview = WKWebView.alloc().initWithFrame_configuration_(content_rect, cfg)
        self._webview.setAutoresizingMask_(18)  # width + height flexible

        url_obj = objc.lookUpClass("NSURL").URLWithString_(GRADIO_URL)
        request = objc.lookUpClass("NSURLRequest").requestWithURL_(url_obj)
        self._webview.loadRequest_(request)

        self._window.setContentView_(self._webview)

        # Delegate
        self._delegate = ARIAWindowDelegate.alloc().init()
        self._delegate.setController_(self)
        self._window.setDelegate_(self._delegate)

        # Register Escape key to hide
        self._register_local_escape()

    def _register_local_escape(self):
        """Add a local event monitor for Escape while our window is key."""
        def escape_handler(event):
            if event.keyCode() == 53:  # Escape
                self.hide_window()
                return None
            return event

        objc.lookUpClass("NSEvent").addLocalMonitorForEventsMatchingMask_handler_(
            1 << 10,  # NSEventMaskKeyDown
            escape_handler,
        )

    def toggle(self):
        """Show or hide the window (called from hotkey handler on main thread)."""
        self._build()
        if self._visible:
            self.hide_window()
        else:
            self.show_window()

    def show_window(self):
        self._build()
        self._window.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)
        self._visible = True

    def hide_window(self):
        if self._window:
            self._window.orderOut_(None)
        self._visible = False


# ---------------------------------------------------------------------------
# Global hotkey via CGEventTap (Option+Space)
# ---------------------------------------------------------------------------

_window_controller: ARIAWindowController = None


def _event_tap_callback(proxy, event_type, event, refcon):
    """CGEventTap callback — runs on a private thread."""
    if event_type == kCGEventKeyDown:
        keycode = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
        flags = CGEventGetFlags(event)
        option_down = bool(flags & kCGEventFlagMaskAlternate)

        if keycode == OPTION_SPACE_KEYCODE and option_down:
            # Dispatch toggle to main thread
            objc.lookUpClass("NSOperationQueue").mainQueue().addOperationWithBlock_(
                lambda: _window_controller.toggle()
            )
            # Consume the event so Option+Space doesn't reach the frontmost app
            return None

    return event


def _install_global_hotkey():
    """Set up the CGEventTap on a background thread with its own run loop."""
    tap = CGEventTapCreate(
        kCGSessionEventTap,
        kCGHeadInsertEventTap,
        0,  # kCGEventTapOptionDefault
        1 << kCGEventKeyDown,
        _event_tap_callback,
        None,
    )

    if tap is None:
        print(
            "[ARIA] WARNING: Could not create event tap.\n"
            "       Grant Accessibility access in System Settings → Privacy & Security → Accessibility.",
            file=sys.stderr,
        )
        return

    src = CFMachPortCreateRunLoopSource(None, tap, 0)
    loop = CFRunLoopGetCurrent()
    CFRunLoopAddSource(loop, src, CoreFoundation.kCFRunLoopCommonModes)
    CGEventTapEnable(tap, True)
    CoreFoundation.CFRunLoopRun()


# ---------------------------------------------------------------------------
# rumps menubar app
# ---------------------------------------------------------------------------

class ARIAMenubarApp(rumps.App):

    def __init__(self, window_controller: ARIAWindowController):
        super().__init__("🧠", quit_button=None)
        self._wc = window_controller
        self.menu = [
            rumps.MenuItem("Open ARIA", callback=self._open),
            None,  # separator
            rumps.MenuItem("Quit", callback=self._quit),
        ]

    @rumps.clicked("Open ARIA")
    def _open(self, _):
        self._wc.show_window()

    @rumps.clicked("Quit")
    def _quit(self, _):
        rumps.quit_application()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _run_intake_if_needed():
    """Run terminal intake and save profile if data/user_profile.json doesn't exist."""
    import pathlib
    profile_path = pathlib.Path(__file__).parent.parent / "data" / "user_profile.json"
    if not profile_path.exists():
        print("[ARIA menubar] No user profile found — running first-launch intake in terminal.")
        _ensure_dependencies()
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from main import run_intake
        from memory.graph import save_profile
        profile = run_intake()
        save_profile(profile)
        print("[ARIA menubar] Profile saved. Starting backend…")


def main():
    # 0. Run intake in main thread if profile doesn't exist yet
    _run_intake_if_needed()

    # 1. Start ARIA backend in a daemon thread
    print("[ARIA menubar] Starting ARIA backend…")
    backend_thread = threading.Thread(target=_start_aria_backend, daemon=True)
    backend_thread.start()

    # 2. Poll until Gradio is ready (up to 60 s)
    print("[ARIA menubar] Waiting for Gradio at localhost:7860…")
    ready = _poll_until_ready(timeout=60)
    if ready:
        print("[ARIA menubar] Gradio is ready.")
    else:
        print("[ARIA menubar] WARNING: Gradio did not respond in 60 s — continuing anyway.")

    # 3. Create shared window controller
    global _window_controller
    _window_controller = ARIAWindowController()

    # 4. Start global hotkey on a background thread (has its own CFRunLoop)
    hotkey_thread = threading.Thread(target=_install_global_hotkey, daemon=True)
    hotkey_thread.start()

    # 5. Launch menubar app (runs the NSApplication main loop on the main thread)
    print("[ARIA menubar] Menubar icon active.  Press Option+Space to summon ARIA.")
    app = ARIAMenubarApp(_window_controller)
    app.run()


if __name__ == "__main__":
    main()
