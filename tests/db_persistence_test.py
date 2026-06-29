from playwright.sync_api import sync_playwright

JD_TEXT = open("data/sample_job_descriptions/backend_engineer_jd.txt").read()

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    page.goto("http://localhost:8520")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1500)

    page.locator("textarea").first.fill(JD_TEXT)
    page.locator('input[type="file"]').first.set_input_files([
        "data/sample_resumes/ahmed_raza.pdf",
        "data/sample_resumes/hina_tariq.docx",
    ])
    page.wait_for_timeout(1000)

    page.get_by_role("button", name="Process & rank candidates").click()
    page.wait_for_selector("text=Processing", state="hidden", timeout=30000)
    page.wait_for_timeout(1500)

    body_text = page.inner_text("body")
    if "Saved to history as job" in body_text:
        print("PASS: save-to-history confirmation message appeared")
    else:
        print("FAIL: no save-to-history confirmation found")
    page.screenshot(path="/tmp/db_test_01_after_processing.png", full_page=True)

    # Now reload the page completely fresh — this simulates closing and
    # reopening the dashboard. If persistence genuinely works, the history
    # panel should show this job even though we re-loaded everything.
    page.goto("http://localhost:8520")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1500)

    history_expander = page.get_by_text("Screening history", exact=False).first
    history_expander.click()
    page.wait_for_timeout(1000)

    body_text_2 = page.inner_text("body")
    if "Backend Software Engineer" in body_text_2 or "Untitled" in body_text_2:
        print("PASS: history panel shows a saved job after full page reload")
    else:
        print("FAIL: history panel does not show the saved job after reload")

    page.screenshot(path="/tmp/db_test_02_history_after_reload.png", full_page=True)

    browser.close()
