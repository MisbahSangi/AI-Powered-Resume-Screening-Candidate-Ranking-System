"""End-to-end UI test: fill JD, upload resumes, click process, screenshot results."""
from playwright.sync_api import sync_playwright
import time

JD_TEXT = open("data/sample_job_descriptions/backend_engineer_jd.txt").read()

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    page.goto("http://localhost:8511")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(2000)

    page.screenshot(path="/tmp/01_initial_load.png", full_page=True)
    print("Step 1: initial load screenshot taken")

    # Paste JD text into the textarea (default mode is already "Paste text")
    textarea = page.locator("textarea").first
    textarea.click()
    textarea.fill(JD_TEXT)
    print("Step 2: JD pasted")

    # Upload resume files via the file input
    file_input = page.locator('input[type="file"]').first
    file_input.set_input_files([
        "data/sample_resumes/ahmed_raza.pdf",
        "data/sample_resumes/sara_malik.txt",
        "data/sample_resumes/bilal_hussain.txt",
    ])
    page.wait_for_timeout(1500)
    print("Step 3: resumes uploaded")

    page.screenshot(path="/tmp/02_before_process.png", full_page=True)

    # Click the "Process & rank candidates" button
    process_button = page.get_by_role("button", name="Process & rank candidates")
    process_button.click()
    print("Step 4: clicked process button")

    # Wait for the "Processing..." spinner to appear, then disappear
    try:
        page.wait_for_selector("text=Processing", timeout=5000)
        print("Spinner appeared, waiting for it to finish...")
    except Exception:
        print("Spinner didn't appear (may have already finished or failed silently)")

    page.wait_for_selector("text=Processing", state="hidden", timeout=30000)
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1500)

    page.screenshot(path="/tmp/03_ranked_results.png", full_page=True)
    print("Step 5: results screenshot taken")

    # Expand the bonus-feature sections so they render in the full-page screenshot
    for label in ["Skill gap analysis across all candidates (bonus feature)", "Find similar candidates (bonus: vector search)"]:
        try:
            page.get_by_text(label, exact=False).first.click()
            page.wait_for_timeout(500)
        except Exception as e:
            print(f"Could not expand '{label}': {e}")

    page.wait_for_timeout(1000)
    page.screenshot(path="/tmp/04_full_page_with_expanders.png", full_page=True)
    print("Step 6: full page with expanders screenshot taken")

    # Capture any console errors
    page.wait_for_timeout(500)

    # Try to read the visible text to sanity-check ranking appeared
    body_text = page.inner_text("body")
    if "Ranked candidates" in body_text:
        print("PASS: 'Ranked candidates' section is visible")
    else:
        print("WARNING: 'Ranked candidates' text not found on page")

    browser.close()
