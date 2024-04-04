import streamlit as st
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent,Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()
api = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="Lyzr Youtube Script Writer",
    layout="centered",  # or "wide"
    initial_sidebar_state="auto",
    page_icon="lyzr-logo-cut.png",
)

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Lyzr Youtube Script Writer")
st.markdown("### Welcome to the Lyzr Youtube Script Writer!")
st.sidebar.markdown("Upload Your Topic Here and get your script ready!!!")

open_ai_text_completion_model = OpenAIModel(
    api_key=api,
    parameters={
        "model": "gpt-4-turbo-preview",
        "temperature": 0.2,
        "max_tokens": 1500,
    },
)
options=["Formal","Casual","Professional","Confident","Simplified","Academic","friendly","funny"]
topic = st.sidebar.text_area("What is the topic of your script?")
objective=st.sidebar.text_input("What is the objective of your script?")
tone = st.sidebar.selectbox("Select a tone of voice", options=options)

if st.sidebar.button("Generate"):
    youtuber_agent = Agent(
        role="Youtube expert",
        prompt_persona=f"You are an Expert YOUTUBE SCRIPTWRITER. Your task is to DEVELOP a COMPELLING and ENGAGING script for an upcoming YouTube video."
    )

    prompt=f"Generate a script covering {topic} in a {tone} manner and objective of script is {objective}. The script should cater to both beginners and advanced viewers."

    script_generation_task  =  Task(
        name="Get Youtube Script",
        model=open_ai_text_completion_model,
        agent=youtuber_agent,
        instructions=prompt,
    )

    youtube_script_title=Task(
        name="Get Youtube Script",
        model=open_ai_text_completion_model,
        agent=youtuber_agent,
        instructions=f"Generate a script title covering {topic} in a {tone} manner and objective of script is {objective}",
    )

    output = LinearSyncPipeline(
            name="Youtube script Generation",
            completion_message="Script Generated",
            tasks=[youtube_script_title,script_generation_task],
    ).run()
    st.markdown(output[0]['task_output'])
