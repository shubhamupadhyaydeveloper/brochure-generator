import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
from scraper import fetch_website_contents, fetch_website_links
import json

st.title("Generate brochure for your company")
st.caption("Note this is a demo, not a final product")

# tools

read_website = {
    "type": "function",
    "function": {
        "name": "fetch_website_contents",
        "description": "Fetch and return the title and text contents of a website. Use this to read the main content of a page.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL of the website to fetch, e.g. https://example.com",
                }
            },
            "required": ["url"],
        },
    },
}

get_all_links = {
    "type": "function",
    "function": {
        "name": "fetch_website_links",
        "description": "Fetch all links from a webpage. Use this to discover relevant pages like About, Careers, Products, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL of the website to fetch links from, e.g. https://example.com",
                }
            },
            "required": ["url"],
        },
    },
}

tools = [read_website, get_all_links]


def handle_tool_calls(message):
    responses = []
    for tool_call in message.tool_calls:
        arguments = json.loads(tool_call.function.arguments)
        url = arguments.get("url")

        if tool_call.function.name == "fetch_website_contents":
            contents = fetch_website_contents(url)
            responses.append(
                {"role": "tool", "content": contents, "tool_call_id": tool_call.id}
            )
        elif tool_call.function.name == "fetch_website_links":
            links = fetch_website_links(url)
            # Convert list to JSON string for the API
            responses.append(
                {
                    "role": "tool",
                    "content": json.dumps(links),
                    "tool_call_id": tool_call.id,
                }
            )
    return responses


load_dotenv(override=True)
google_api_key = os.getenv("GOOGLE_API_KEY")
gemini = OpenAI(
    api_key=google_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

system_message = """You are an expert brochure writer. When a user provides a company website URL:

1. First, use fetch_website_links to get all links from the website
2. Identify relevant pages (About, Products, Services, Careers, Contact) and format them as:
   {
       "links": [
           {"type": "about page", "url": "https://full.url/about"},
           {"type": "careers page", "url": "https://full.url/careers"}
       ]
   }
3. Use fetch_website_contents to read the main page and relevant subpages
4. Create a professional marketing brochure with:
   - Company Overview
   - Key Products/Services
   - Why Choose Them (unique value proposition)
   - Contact Information

Use markdown formatting. Only help with brochure generation - politely decline other requests."""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi there! Enter your company URL and I'll generate a professional brochure for you.",
        }
    ]

if "brochure" not in st.session_state:
    st.session_state.brochure = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Show download button if brochure exists
if st.session_state.brochure:
    st.download_button(
        label="Download Brochure",
        data=st.session_state.brochure,
        file_name="company_brochure.md",
        mime="text/markdown",
    )

if prompt := st.chat_input("Enter your company URL"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Build message history properly
        messages = [{"role": "system", "content": system_message}]
        messages.extend(
            [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
        )

        placeholder = st.empty()
        tools_were_used = False

        with st.status("Processing...", expanded=True) as status:
            st.write("Analyzing your request...")
            response = gemini.chat.completions.create(
                model="gemini-2.5-flash-lite",
                messages=messages,
                tools=tools,
            )

            # Handle tool calls loop
            while response.choices[0].finish_reason == "tool_calls":
                tools_were_used = True
                assistant_message = response.choices[0].message
                messages.append(assistant_message)

                # Show which tools are being called
                for tool_call in assistant_message.tool_calls:
                    if tool_call.function.name == "fetch_website_contents":
                        st.write("Fetching website content...")
                    elif tool_call.function.name == "fetch_website_links":
                        st.write("Discovering relevant pages...")

                tool_responses = handle_tool_calls(assistant_message)
                messages.extend(tool_responses)

                st.write("Generating brochure...")
                response = gemini.chat.completions.create(
                    model="gemini-2.5-flash-lite",
                    messages=messages,
                    tools=tools,
                )

            if tools_were_used:
                status.update(label="Brochure complete!", state="complete", expanded=False)
            else:
                status.update(label="Done", state="complete", expanded=False)

        # Get final response content
        final_content = response.choices[0].message.content
        placeholder.markdown(final_content)

        # Only store brochure if tools were used (actual brochure generated)
        if tools_were_used:
            st.session_state.brochure = final_content

        st.session_state.messages.append(
            {"role": "assistant", "content": final_content}
        )

        # Rerun to show download button if brochure was generated
        if tools_were_used:
            st.rerun()
