# Screenshots

The images here are captures of the real dashboard served by a real coordinator,
not mockups. `dashboard-desktop.png` is 1440x1220 at 2x, `dashboard-phone.png` is
430x1500 at 2x.

Regenerate them:

```bash
make screenshots
```

That runs [`tools/cpp_port/grid/screenshots.sh`](../../tools/cpp_port/grid/screenshots.sh),
which:

1. builds the coordinator, worker and fixture trainer if they are missing
2. starts a coordinator on a random port with a scratch database
3. submits two short jobs and lets five workers finish them
4. starts one worker on a trainer that holds its replica open, then submits a
   longer job, so the page shows a job in flight
5. captures `/` with headless Chrome at both widths

Needs `google-chrome` or `chromium` on `PATH`. Set `SCREENSHOT_PORT` to pin the
port and `SCREENSHOT_DIR` to write elsewhere.

If the layout changes, look at the fresh captures rather than trusting the
numbers in the README: the phone capture is the one that catches overflow, and
the desktop one catches a table that grew a column and now wraps.
