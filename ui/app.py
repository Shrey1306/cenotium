"""Streamlit UI for Cenotium."""

import streamlit as st
from streamlit_option_menu import option_menu


def set_custom_css():
    st.markdown(
        """
        <style>
            body, p, li { font-size: 20px !important; }
            h1 { font-size: 34px !important; }
            h2 { font-size: 28px !important; }
            h3 { font-size: 24px !important; }
        </style>
    """,
        unsafe_allow_html=True,
    )


def home():
    st.markdown(
        "<h1 style='text-align: center;'>Cenotium: Agentic Internet Browser</h1>",
        unsafe_allow_html=True,
    )

    st.write("""
    The internet is evolving into a dynamic ecosystem where AI agents operate
    independently, transforming the way tasks are executed, decisions are made,
    and digital interactions take place.

    This Agentic Internet is a fundamental shift toward a world where autonomous
    agents handle information retrieval, transactions, security, and complex
    problem-solving with minimal human input.

    However, today's websites are built for human users, not AI agents. We have
    built a browser designed exclusively for AI agents, reimagining the way they
    access and interact with online information.
    """)


def scraper_page():
    st.markdown(
        "<h1 style='text-align: center;'>Web Schema Development</h1>",
        unsafe_allow_html=True,
    )

    st.write("""
    The current internet is not designed for AI agents. Websites rely on front-end
    libraries to render content dynamically, making UI elements inaccessible to
    autonomous systems without structured interfaces.

    To solve this, we created a custom schema layer that transforms standard web
    pages into structured, interactable formats that AI agents can understand.
    """)

    st.subheader("1. Planning Model")
    st.write("Identifies interactive elements using vision-based AI models.")

    st.subheader("2. Grounding Model (OS-Atlas)")
    st.write("Maps interaction coordinates using spatial-aware AI models.")

    st.subheader("3. Action Model")
    st.write("Executes interactions with identified elements.")


def agents_page():
    st.markdown(
        "<h1 style='text-align: center;'>Agent Architecture</h1>",
        unsafe_allow_html=True,
    )

    st.write("""
    The Agent Manager is the central hub for receiving prompts, interpreting them,
    and orchestrating AI-powered agents to execute tasks autonomously.
    """)

    st.subheader("Available Agents")

    st.write(
        "**Perplexity Search Agent**: Real-time web searches and knowledge retrieval."
    )
    st.write(
        "**Twilio Calling Agent**: Automated outbound calls with AI-generated messages."
    )
    st.write(
        "**Browser Activation Agent**: Dynamic web page interaction and automation."
    )


def security_page():
    st.markdown(
        "<h1 style='text-align: center;'>Security and Trust</h1>",
        unsafe_allow_html=True,
    )

    st.write("""
    Our security infrastructure ensures agent-to-agent interactions, data integrity,
    and trust validation at scale.
    """)

    st.subheader("Trust System")
    st.write("""
    - Local Trust: Direct agent-to-agent interactions
    - Global Trust: Network-wide reputation aggregation
    - Temporal Decay: Recent transactions weighted higher
    """)

    st.subheader("Security Protocols")
    st.write("""
    - Fernet Symmetric Encryption (AES-CBC + HMAC-SHA256)
    - Rate Limiting and Digital Signatures
    - Secure Inter-Agent Communication
    """)


def main():
    st.set_page_config(layout="wide")
    set_custom_css()

    with st.sidebar:
        selected = option_menu(
            "Menu",
            ["Home", "Web Schema", "Agents", "Security"],
            icons=["house", "search", "robot", "shield"],
            menu_icon="menu-hamburger",
            default_index=0,
        )

    if selected == "Home":
        home()
    elif selected == "Web Schema":
        scraper_page()
    elif selected == "Agents":
        agents_page()
    elif selected == "Security":
        security_page()


if __name__ == "__main__":
    main()
