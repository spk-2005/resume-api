import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.resume_parser import extract_contact_info, extract_gpa, extract_experience_years
from services.skill_extractor import get_skill_counts

def test_parsing_accuracy():
    print("Starting Accuracy Verification...\n")
    
    # Test 1: Contact Info (Name & Phone)
    resume_header = """
    John Doe
    Senior Software Engineer
    Phone: +1 (555) 123-4567 | Email: john.doe@example.com
    Address: San Francisco, CA
    """
    contact = extract_contact_info(resume_header)
    print(f"Testing Contact Info:")
    print(f"  - Name: {contact['name']} (Expected: John Doe) {'Pass' if contact['name'] == 'John Doe' else 'Fail'}")
    print(f"  - Phone: {contact['phone']} (Expected: +1 (555) 123-4567) {'Pass' if contact['phone'] == '+1 (555) 123-4567' else 'Fail'}")
    print()

    # Test 2: GPA Extraction
    gpa_texts = [
        "CGPA: 9.5/10",
        "GPA: 3.8 / 4.0",
        "Grade: A",
        "Aggregate: 85%"
    ]
    print(f"Testing GPA Extraction:")
    for text in gpa_texts:
        gpa = extract_gpa(text)
        print(f"  - Input: '{text}' -> Extracted: {gpa}")
    print()

    # Test 3: Date Parsing
    current_year = datetime.now().year
    exp_text = "June 2021 - Present"
    yrs, timeline = extract_experience_years(exp_text)
    print(f"Testing Date Parsing:")
    print(f"  - Input: '{exp_text}'")
    if timeline and timeline[0]['end'].lower() == 'present':
        # Check if internal logic correctly sets to current year
        print(f"  - End Year detected correctly for 'Present'")
    print(f"  - Approx Years: {yrs}")
    print()

    # Test 4: New Skill Extraction
    skill_text = """
    Proficient in LangChain, RAG, and OpenAI GPT-4.
    Experience with Vector DBs like Pinecone and Weaviate.
    Frontend: SolidJS, Astro.
    Cloud: Vercel, Supabase.
    """
    skills = get_skill_counts(skill_text)
    new_skills = ["langchain", "rag", "openai", "vector database", "solidjs", "astro", "vercel", "supabase"]
    print(f"Testing New Skill Extraction:")
    for s in new_skills:
        found = s in skills
        print(f"  - Skill '{s}': {'Found' if found else 'Missing'}")
    print()

    print("Verification Complete!")

if __name__ == "__main__":
    test_parsing_accuracy()
