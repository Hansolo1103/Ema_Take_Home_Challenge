import streamlit as st
from llm_chain import load_normal_chain, load_pdf_chat_chain
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import yaml
import os
from html_templates import css, bot_template, user_template
from utils import save_chat_history_json, get_timestamp, load_chat_history_json
from streamlit_mic_recorder import mic_recorder
from audio_handler import transcribe_audio
from image_handler import handle_image
from pdf_handler import add_docs_to_db

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def load_chain(chat_history):
    if st.session_state.pdf_chat:
        return load_pdf_chat_chain(chat_history)
    return load_normal_chain(chat_history)


def clear_input_field():
    st.session_state.user_question = st.session_state.user_input
    st.session_state.user_input = ""


def set_send_input():
    st.session_state.send_input = True
    clear_input_field()


def toggle_pdf_chat():
    st.session_state.pdf_chat = True


def save_chat_history():
    if st.session_state.history != []:
        if st.session_state.session_key == "new_session":
            st.session_state.new_session_key = get_timestamp() + ".json"
            save_chat_history_json(st.session_state.history,
                                   config["chat_history_path"] + st.session_state.new_session_key)
        else:
            save_chat_history_json(st.session_state.history,
                                   config["chat_history_path"] + st.session_state.session_key + ".json")


def track_index():
    st.session_state.session_index_tracker = st.session_state.session_key


def main():
    st.title('Multimodal Local Chat App')
    st.write(css, unsafe_allow_html=True)

    chat_container = st.container()
    st.sidebar.title('Chat Sessions')

    chat_sessions = ["new_session"] + os.listdir(config["chat_history_path"])

    if "send_input" not in st.session_state:
        st.session_state.session_key = "new_session"
        st.session_state.send_input = False
        st.session_state.user_question = ""
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "new_session"

    if st.session_state.session_key == "new_session" and st.session_state.new_session_key != None:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None

    index = chat_sessions.index(st.session_state.session_index_tracker)
    st.sidebar.selectbox("Select a chat session",
                         chat_sessions, key="session_key", index=index, on_change=track_index)

    st.sidebar.toggle("PDF Chat", key="pdf_chat", value=False)

    if st.session_state.session_key != "new_session":
        st.session_state.history = load_chat_history_json(
            config["chat_history_path"] + st.session_state.session_key)
    else:
        st.session_state.history = []

    chat_history = StreamlitChatMessageHistory(key="history")
    llm_chain = load_chain(chat_history)

    user_input = st.text_input(
        "Type message here", key="user_input", on_change=set_send_input)

    voice_recording_column, send_button_column = st.columns(2)

    with voice_recording_column:
        voice_recording = mic_recorder(
            start_prompt="Start recording", stop_prompt="Stop recording")

    with send_button_column:
        send_button = st.button("Send", key="send_button",
                                on_click=clear_input_field)

    uploaded_audio = st.sidebar.file_uploader(
        "Upload an audio file", type=["wav", "mp3", "ogg"])

    uploaded_image = st.sidebar.file_uploader(
        "Upload an image file", type=["jpg", "jpeg", "png"])

    uploaded_pdf_file = st.sidebar.file_uploader(
        "Upload a document", accept_multiple_files=True, type=["pdf"],  on_change=toggle_pdf_chat)

    if uploaded_pdf_file:
        with st.spinner("Processing PDF..."):
            add_docs_to_db(uploaded_pdf_file)

    if uploaded_audio:
        transcribed_audio = transcribe_audio(uploaded_audio.getvalue())
        llm_chain.run("Summarize this audio: " + transcribed_audio)

    if voice_recording:
        transcribed_audio = transcribe_audio(voice_recording["bytes"])
        print(transcribed_audio)
        llm_chain.run(transcribed_audio)

    if send_button or st.session_state.send_input:
        if uploaded_image:
            with st.spinner("Processing ..."):
                user_message = st.session_state.user_question if st.session_state.user_question != "" else "Describe the image"
                st.session_state.user_question = ""

                llm_answer = handle_image(
                    uploaded_image.getvalue(), user_message)
                chat_history.add_user_message(user_message)
                chat_history.add_ai_message(llm_answer)

                st.image(uploaded_image, caption="Uploaded Image",
                         use_column_width=True)
                st.chat_message("ai").write(llm_answer)

        if st.session_state.user_question != "":
            st.chat_message("user").write(st.session_state.user_question)
            llm_response = llm_chain.run(st.session_state.user_question)
            st.chat_message("ai").write(llm_response)
            st.session_state.user_question = ""

    if chat_history.messages != []:
        with chat_container:
            st.write("Chat History")
            for message in chat_history.messages:

                if message.type == "human":
                    st.write(user_template.replace("{{MSG}}", message.content))

                else:
                    st.write(bot_template.replace("{{MSG}}", message.content))
                st.chat_message(message.type).write(message.content)

    save_chat_history()


if __name__ == "__main__":
    main()
