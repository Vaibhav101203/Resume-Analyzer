import re
import nltk
import time
import random
import subprocess
from PIL import Image
import streamlit as st
from PyPDF2 import PdfReader
from streamlit_tags import st_tags

from skills import skill_list
from skills import ds_list
from skills import web_list
from skills import ios_list
from skills import android_list
from skills import uiux_list
from skills import languages

from courses import ds_course
from courses import web_course
from courses import android_course
from courses import ios_course
from courses import uiux_course
from courses import interview_videos

nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("maxent_ne_chunker", quiet=True)
nltk.download("words", quiet=True)
nltk.download("stopwords", quiet=True)


def extract_data(pdf_path):
    reader = PdfReader(pdf_path)
    pages = len(reader.pages)
    page = reader.pages[0]
    text = page.extract_text()
    return text, pages


def extract_names(txt):
    person_names = []

    for sent in nltk.sent_tokenize(txt):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, "label") and chunk.label() == "PERSON":
                person_names.append(
                    " ".join(chunk_leave[0] for chunk_leave in chunk.leaves())
                )

    return person_names


PHONE_REG = re.compile(r"[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]")


def doc_to_text_catdoc(file_path):
    try:
        process = subprocess.Popen(
            ["catdoc", "-w", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    except (
        FileNotFoundError,
        ValueError,
        subprocess.TimeoutExpired,
        subprocess.SubprocessError,
    ) as err:
        return (None, str(err))
    else:
        stdout, stderr = process.communicate()

    return (stdout.strip(), stderr.strip())


def extract_phone_number(resume_text):
    phone = re.findall(PHONE_REG, resume_text)

    if phone:
        number = "".join(phone[0])

        if resume_text.find(number) >= 0 and len(number) < 30:
            return number
    return None


EMAIL_REG = re.compile(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+")


def extract_emails(resume_text):
    return re.findall(EMAIL_REG, resume_text)


def skill_area(job_applied):
    if job_applied == "data scientist":
        return ds_list
    if job_applied == "web developer":
        return web_list
    if job_applied == "android developer":
        return android_list
    if job_applied == "ios developer":
        return ios_list
    if job_applied == "uiux developer":
        return uiux_list
    else:
        print("Please Enter a valid Post")


def extract_skills(input_text, job_applied):
    stop_words = set(nltk.corpus.stopwords.words("english"))
    word_tokens = nltk.tokenize.word_tokenize(input_text)

    filtered_tokens = [w for w in word_tokens if w not in stop_words]

    filtered_tokens = [w for w in word_tokens if w.isalpha()]

    bigrams_trigrams = list(map(" ".join, nltk.everygrams(filtered_tokens, 2, 3)))

    found_skills = set()

    for token in filtered_tokens:
        if token.lower() in skill_area(job_applied):
            found_skills.add(token)

    for ngram in bigrams_trigrams:
        if ngram.lower() in skill_area(job_applied):
            found_skills.add(ngram)

    return found_skills


RESERVED_WORDS = [
    "school",
    "college",
    "university",
    "academy",
    "faculty",
    "institute",
]


def extract_education(input_text):
    organizations = []

    for sent in nltk.sent_tokenize(input_text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, "label") and chunk.label() == "ORGANIZATION":
                organizations.append(" ".join(c[0] for c in chunk.leaves()))

    # we search for each bigram and trigram for reserved words
    # (college, university etc...)
    education = set()
    for org in organizations:
        for word in RESERVED_WORDS:
            if org.lower().find(word) >= 0:
                education.add(org)

    return education


def extract_Resume_Score(input_text, skills):
    stop_words = set(nltk.corpus.stopwords.words("english"))
    word_tokens = nltk.tokenize.word_tokenize(input_text)

    filtered_tokens = [w for w in word_tokens if w not in stop_words]

    filtered_tokens = [w for w in word_tokens if w.isalpha()]

    filtered_tokens = set(filtered_tokens)

    bigrams_trigrams = list(map(" ".join, nltk.everygrams(filtered_tokens, 2, 3)))

    score = 0

    filtered_tokens = [element.lower() for element in filtered_tokens]

    cand_lvl = "Fresher"

    score += len(skills) * 2.8

    if "experiance" in filtered_tokens:
        score += 6
    if "intership" or "interships" in filtered_tokens:
        score += 12
        cand_lvl = "Intermediate"
    if "achivements" in filtered_tokens:
        score += 10
    if "certification" or "certifications" in filtered_tokens:
        score += 6
    if "projects" or "project" in filtered_tokens:
        score += 10

    for ngram in bigrams_trigrams:
        if ngram.lower() == "work experiance":
            cand_lvl = "Experianced"

    return score, cand_lvl


def language_(input_text):
    stop_words = set(nltk.corpus.stopwords.words("english"))
    word_tokens = nltk.tokenize.word_tokenize(input_text)

    filtered_tokens = [w for w in word_tokens if w not in stop_words]

    filtered_tokens = [w for w in word_tokens if w.isalpha()]

    filtered_tokens = set(filtered_tokens)

    language = set()

    for token in filtered_tokens:
        if token.lower() in languages:
            language.add(token)

    return language


def recommend(job_applied):
    if job_applied == "data scientist":
        return ds_course
    if job_applied == "web developer":
        return web_course
    if job_applied == "android developer":
        return android_course
    if job_applied == "ios developer":
        return ios_course
    if job_applied == "uiux developer":
        return uiux_course
    else:
        return None


def course_recommender(course_list):
    st.subheader("**Courses & Certificates Recommendations 👨‍🎓**")
    c = 0
    rec_course = []
    no_of_reco = st.slider("Choose Number of Course Recommendations:", 1, 10, 5)
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course


def run():
    img = Image.open("RESUME.png")
    st.title("Resume Analyzer")
    st.image(img, caption="", use_column_width=True)
    activities = ["# Choose ", "User", "About"]
    choice = st.sidebar.selectbox("Choose among the given options:", activities)

    if choice == "# Choose":
        st.write("Please select any option")

    elif choice == "User":
        act_name = st.text_input("Name*")
        act_mail = st.text_input("Mail*")
        job_option = skill_list
        act_job = st.selectbox("Job*", job_option)

        st.markdown(
            """<h5 style='text-align: left; color: #021659;'> Upload Your Resume, And Get Smart Recommendations</h5>""",
            unsafe_allow_html=True,
        )
        pdf_file = st.file_uploader("Choose your resume", type=["pdf"])
        if pdf_file is not None:
            with st.spinner("Analyzing Resume ... "):
                time.sleep(6)

        path = pdf_file
        text, pages = extract_data(path)
        job_applied = act_job
        education_info = extract_education(text)
        names = extract_names(text)
        skills = extract_skills(text, job_applied)
        emails = extract_emails(text)
        literate = extract_education(text)
        phone_number = extract_phone_number(text)
        score, cand_lvl = extract_Resume_Score(text, skills)
        video_recommend = recommend(job_applied)
        languages = language_(text)

        st.markdown(
            f"""<h4 style='text-align: left; color: #d73b5c;'>Your Basic Info</h4>""",
            unsafe_allow_html=True,
        )
        st.text("Name: " + act_name)
        st.text("Email: " + emails[0])
        st.text("Phone Number: " + phone_number)

        if languages is not None:
            languages = list(languages)
            st_tags(
                label="### Programming Languages", text="", value=languages, key="1 "
            )

        if skills is not None:
            skills = list(skills)
            st_tags(label="### Your Current Skills", text="", value=skills, key="1  ")

        if score:
            st.markdown(
                f"""<h4 style='text-align: left; color: #d73b5c;'>You Score is: {score}/100!</h4>""",
                unsafe_allow_html=True,
            )

        if cand_lvl == "Fresher":
            st.markdown(
                """<h4 style='text-align: left; color: #d73b5c;'>You are at Fresher level!</h4>""",
                unsafe_allow_html=True,
            )

        elif cand_lvl == "Experianced":
            st.markdown(
                """<h4 style='text-align: left; color: #d73b5c;'>You are at Experianced level!</h4>""",
                unsafe_allow_html=True,
            )

        elif cand_lvl == "Intermediate":
            st.markdown(
                """<h4 style='text-align: left; color: #d73b5c;'>You are at Intermidate level!</h4>""",
                unsafe_allow_html=True,
            )

        if pdf_file:
            rec = ["", "Yes", "No"]
            reco = st.selectbox("Recommendation", rec)

            if reco == "Yes":
                recommended_skills = skill_area(job_applied)
                recommended_keywords = st_tags(
                    label="### Recommended skills for you.",
                    text="Recommended skills generated from System",
                    value=recommended_skills,
                    key="2",
                )
                st.markdown(
                    """<h5 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job</h5>""",
                    unsafe_allow_html=True,
                )
                # course recommendation
                course = recommend(job_applied)
                rec_course = course_recommender(course)

            if reco == "No":
                st.text("Thanks !!")

    elif choice == "About":
        st.markdown(
            """<h4 style='text-align: left; color: #d73b5c;'>About Us</h4>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<h6 style='text-align: left; color: #fffdd0;'>Welcome to our resume analyzing app, created by Vaibhav Yadav and Tejas Sharma!</h6>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<h4 style='text-align: left; color: #d73b5c;'>Our Mission</h4>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<h6 style='text-align: left; color: #fffdd0;'>Our mission is to simplify the job application process by providing a powerful tool to analyze resumes quickly and efficiently. We understand that crafting the perfect resume can be challenging, and our app aims to assist job seekers in optimizing their resumes to stand out in the competitive job market.</h6>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<h4 style='text-align: left; color: #d73b5c;'>What We Offer</h4>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<h6 style='text-align: left; color: #fffdd0;'>Our app offers comprehensive analysis of resumes, including skills assessment, education evaluation, and scoring based on relevant experience and achievements. With our intuitive interface and advanced algorithms, users can gain valuable insights into their resumes to enhance their chances of success in landing their dream job.</h6>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<h4 style='text-align: left; color: #d73b5c;'>Our Team</h4>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<h6 style='text-align: left; color: #fffdd0;'>Vaibhav Yadav</h6>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<h6 style='text-align: left; color: #fffdd0;'>Tejas Sharma</h6>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<h4 style='text-align: left; color: #d73b5c;'>Contact Us</h4>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<h6 style='text-align: left; color: #fffdd0;'>Vaibhav <br>Phone Number: +91 8917-045-350  <br>Email: vaibhavyangkai123@gmail.com  <br>Github: Vaibhav101203</h6>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<h6 style='text-align: left; color: #fffdd0;'>Tejas <br>Phone Number: +91 7727-048-420  <br>Email: tsharma1704@gmail.com  <br>Github: Tej-as1</h6>""",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    run()
