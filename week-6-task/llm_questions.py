from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

questions = [

    "Explain the benefits and challenges of being a student at UC Santa Cruz.",

    "Describe how technology is used in university education.",

    "Summarize the importance of internships for computer science students.",

    """
Rules:
All UCSC students who complete CSE101 meet the programming requirement.
Srithija completed CSE101.

Example:
Sofia completed CSE101 -> Meets programming requirement

Question:
Does Srithija meet the programming requirement?
""",

    """
Rules:
Students with a GPA above 3.0 qualify for the honors program.
Josh has a GPA of 3.5.

Example:
Carlos has a GPA of 3.8 -> Qualifies

Question:
Does Josh qualify for the honors program?
""",

    """
Rules:
All students enrolled in CSE150 learn computer networking concepts.
Zoro is enrolled in CSE150.

Example:
David is enrolled in CSE150 -> Learns networking concepts

Question:
Does Zoro learn computer networking concepts?
""",

    """
Rules:
All UCSC students have a student ID.
Priya is a UCSC student.

Question:
Does Priya have a student ID?

Answer only Yes or No.
""",

    """
Rules:
Students who earn more than 180 credits can graduate.
Miguel has earned 200 credits.

Question:
Can Miguel graduate?

Answer only True or False.
""",

    """
Rules:
All students enrolled in CSE101 are taking a programming course.
Suga is enrolled in CSE101.

Question:
Is Suga taking a programming course?

Answer only True or False.
"""
]

for i, question in enumerate(questions, start=1):

    print("=" * 70)
    print(f"QUESTION {i}")
    print(question)

    response = generator(
        f"Answer the following question:\n{question}\nAnswer:",
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )

    print("\nANSWER:")
    print(response[0]["generated_text"])
    print("\n")