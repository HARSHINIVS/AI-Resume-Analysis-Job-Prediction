import pandas as pd
import random

roles = {
    "Data Scientist": ["python", "machine learning", "pandas", "numpy", "statistics"],
    "Web Developer": ["html", "css", "javascript", "react", "node", "python"],
    "Software Engineer": ["java", "c++", "algorithms", "data structures", "oop", "python"]
}

noise_words = [
    "team player", "hardworking", "quick learner",
    "communication", "leadership", "management"
]

data = []

for role, skills in roles.items():
    for _ in range(100):

        sample_skills = random.sample(skills, random.randint(1, len(skills)))

        text = " ".join(sample_skills + random.sample(noise_words, 2))

        # Add noise
        if random.random() > 0.5:
            text += " random text analysis business"

        # Add wrong skills
        if random.random() > 0.7:
            text += " html css javascript"

        # Shuffle
        words = text.split()
        random.shuffle(words)
        text = " ".join(words)

        data.append([text, role])

df = pd.DataFrame(data, columns=["resume_text", "role"])
df.to_csv("resume.csv", index=False)

print("Dataset Generated!")