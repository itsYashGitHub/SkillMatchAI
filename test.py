from matcher import calculate_match

resume = "Python, Machine Learning, Data Analysis, SQL"
job = "Looking for a Python developer with ML and SQL skills"

score = calculate_match(resume, job)
print(f"Match Score: {score}%")
