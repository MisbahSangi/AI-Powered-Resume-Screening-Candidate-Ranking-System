from playwright.sync_api import sync_playwright

JD_TEXT = open("data/sample_job_descriptions/backend_engineer_jd.txt").read()
SECOND_JD = """Job Title: Frontend Developer
Requirements:
- 1+ years of experience
- Strong proficiency in JavaScript and React
- Experience with HTML and CSS
- Familiarity with Git
"""

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1400, "height": 1200})
    page.goto("http://localhost:8533")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1500)

    page.locator("textarea").first.fill(JD_TEXT)
    page.locator('input[type="file"]').first.set_input_files([
        "data/sample_resumes/ahmed_raza.pdf",
        "data/sample_resumes/sara_malik.txt",
        "data/sample_resumes/bilal_hussain.txt",
        "data/sample_resumes/hina_tariq.docx",
    ])
    page.wait_for_timeout(1000)
    page.get_by_role("button", name="Process & rank candidates").click()
    page.wait_for_selector("text=Processing", state="hidden", timeout=30000)
    page.wait_for_timeout(2500)
    print("Step 1: processed 4 candidates")

    # Regression check: changing the candidate-detail dropdown used to wipe
    # the whole page back to the empty state (button-gated rendering bug).
    page.get_by_text("Select a candidate", exact=False).scroll_into_view_if_needed()
    target = page.locator(
        "xpath=//*[contains(text(),'Select a candidate')]/following::div[@data-baseweb='select'][1]"
    )
    target.click(timeout=10000)
    page.wait_for_timeout(500)
    page.keyboard.press("ArrowDown")
    page.keyboard.press("Enter")
    page.wait_for_timeout(1500)
    body_after_dropdown = page.inner_text("body")
    if "Ranked candidates" in body_after_dropdown and "Candidate detail" in body_after_dropdown:
        print("PASS: changing candidate dropdown did NOT wipe the results page")
    else:
        print("FAIL: changing candidate dropdown wiped the page (regression!)")
    page.screenshot(path="/tmp/bonus_00_after_dropdown_change.png", full_page=True)

    # Expand clustering
    page.get_by_text("Group candidates into clusters", exact=False).first.click()
    page.wait_for_timeout(1000)
    body = page.inner_text("body")
    if "Cluster 1" in body:
        print("PASS: clustering produced output")
    else:
        print("FAIL: no cluster output found")
    page.screenshot(path="/tmp/bonus_01_clustering.png", full_page=True)

    # Expand multi-job matching, paste second JD, click re-score
    page.get_by_text("Match these same candidates against a different job", exact=False).first.click()
    page.wait_for_timeout(800)
    second_jd_box = page.locator("textarea").last
    second_jd_box.fill(SECOND_JD)
    page.get_by_role("button", name="Re-score against this job").click()
    page.wait_for_timeout(2000)
    page.screenshot(path="/tmp/bonus_02_multijob.png", full_page=True)

    body2 = page.inner_text("body")
    if "Sara Malik" in body2 and "Recommendation" in body2:
        print("PASS: multi-job re-scoring rendered a results table")
    else:
        print("FAIL: multi-job re-scoring did not render expected content")

    browser.close()
