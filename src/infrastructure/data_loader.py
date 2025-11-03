from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import pandas as pd

driver = webdriver.Chrome()
url = "https://fbref.com/en/comps/9/2024-2025/2024-2025-Premier-League-Stats"
driver.get(url)

# Wait for page to load
time.sleep(3)

table = driver.find_element(By.ID, "results2024-202591_overall")

rows = table.find_elements(By.CSS_SELECTOR, "tbody tr")

data = []

for row in rows:
    cells = row.find_elements(By.TAG_NAME, "td")
    if not cells:
        continue
    team_name = row.find_element(By.CSS_SELECTOR, '[data-stat="team"] a').text
    row_data = {
        "Team": team_name,
        "Games": row.find_element(By.CSS_SELECTOR, '[data-stat="games"]').text,
        "Wins": row.find_element(By.CSS_SELECTOR, '[data-stat="wins"]').text,

        "Ties": row.find_element(By.CSS_SELECTOR, '[data-stat="ties"]').text,
        "Losses": row.find_element(By.CSS_SELECTOR, '[data-stat="losses"]').text,
        "Goals For": row.find_element(By.CSS_SELECTOR, '[data-stat="goals_for"]').text,
        "Goals Against": row.find_element(By.CSS_SELECTOR, '[data-stat="goals_against"]').text,
        "Goal Diff": row.find_element(By.CSS_SELECTOR, '[data-stat="goal_diff"]').text,
        "Points": row.find_element(By.CSS_SELECTOR, '[data-stat="points"]').text,
        "xG For": row.find_element(By.CSS_SELECTOR, '[data-stat="xg_for"]').text,

        "xG Against": row.find_element(By.CSS_SELECTOR, '[data-stat="xg_against"]').text,
        "Attendance per Game": row.find_element(By.CSS_SELECTOR, '[data-stat="attendance_per_g"]').text,
        "Top Scorer": row.find_element(By.CSS_SELECTOR, '[data-stat="top_team_scorers"]').text
    }

    data.append(row_data)

driver.quit()

df = pd.DataFrame(data)


df.head()

