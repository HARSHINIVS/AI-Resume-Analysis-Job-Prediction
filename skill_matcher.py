roles_skills = {
    "Data Scientist": ["python", "machine learning", "pandas", "numpy", "statistics"],
    "Web Developer": ["html", "css", "javascript", "react", "node"],
    "Software Engineer": ["java", "c++", "algorithms", "data structures", "oop"]
}

def match_skills(resume_text, role):
    required_skills = roles_skills[role]
    
    found_skills = [skill for skill in required_skills if skill in resume_text]
    missing_skills = list(set(required_skills) - set(found_skills))

    score = int((len(found_skills) / len(required_skills)) * 100)

    return score, missing_skills