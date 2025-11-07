import streamlit as st
import os
import json
from typing import List
from pydantic import BaseModel, Field

# Import the Google GenAI SDK
from google import genai
from google.genai import types
from google.genai.errors import APIError

from fpdf import FPDF
from io import BytesIO # To handle the PDF file in memory

# ==============================================================================
# 1. Configuration and Setup
# ==============================================================================

st.set_page_config(
    page_title="AI Event Planner & Budget Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Data Structure (Pydantic) ---

class Item(BaseModel):
    """Represents a single item in the plan."""
    name: str = Field(description="Name of the item (e.g., 'Balloon Arch', 'Chicken Entree', 'Venue Rental').")
    category: str = Field(description="Category (e.g., 'Decorations', 'Menu', 'Venue', 'Entertainment').")
    estimated_cost_usd: float = Field(description="The estimated cost for this item. MUST be a numerical value.")
    notes: str = Field(description="Action item, local vendor suggestion, or estimated price range for the item.")

class EventPlan(BaseModel):
    """The final, structured output for the entire event plan."""
    theme: str
    timeline_summary: str = Field(description="A brief, 3-step summary of the event flow (e.g., 'Setup -> Ceremony -> Reception').")
    total_budget: float
    items: List[Item]

# ==============================================================================
# 3. Core AI Planning Function (USING GEMINI LLM)
# ==============================================================================

@st.cache_data(show_spinner="Calling Gemini LLM to generate your structured event plan...")
def generate_plan_with_llm(api_key: str, event_type: str, budget: float, people: int, theme: str) -> EventPlan:
    """
    Uses the Gemini LLM and structured output to generate the event plan.
    Uses st.cache_data to prevent re-running the API call unnecessarily.
    """
    try:
        # Initialize the client with the user-provided or secret API key
        client = genai.Client(api_key=api_key)
    except Exception as e:
        raise ConnectionError(f"Error initializing Gemini Client: {e}")

    # --- Construct the detailed prompt for the LLM ---
    prompt = f"""
    You are an expert event planner and budget assistant.
    Your task is to create a comprehensive event plan that strictly adheres to the budget.

    **Event Details:**
    - Primary Event Type: {event_type}
    - Theme: {theme}
    - Total Budget: ${budget:,.2f}
    - Estimated Guests: {people}

    **Planning Instructions:**
    1.  Create a realistic budget breakdown using the Item list, ensuring the sum of 'estimated_cost_usd' **DOES NOT EXCEED** the Total Budget. If there's a remainder, create a 'Contingency/Buffer' item.
    2.  Allocate appropriate funds for categories like Venue, Menu, Decorations, and Entertainment based on the event type ({event_type}) and guest count ({people}).
    3.  The 'notes' field MUST include a specific action, vendor suggestion, or price range relevant to the item.
    4.  Generate a brief, realistic 3-step timeline summary for the event flow.
    """

    # --- Call the Gemini Model with Pydantic Schema for Structured Output ---
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=EventPlan,
            ),
        )

        plan_data = EventPlan.parse_raw(response.text)
        return plan_data

    except APIError as e:
        st.error(f"🛑 Gemini API Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"🛑 Error during LLM call or data parsing: {e}")
        st.stop()

# ==============================================================================
# 4. PDF Generator Function (Adapted for Streamlit)
# ==============================================================================

def generate_pdf(plan: EventPlan) -> BytesIO:
    """Creates a PDF schedule and shopping list from the EventPlan in memory."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    
    # Title
    pdf.cell(0, 10, f"AI Event Plan: {plan.theme}", 0, 1, "C")
    pdf.set_font("Arial", "", 12)
    pdf.ln(5)

    # Summary and Budget
    pdf.cell(0, 8, f"Total Budget: ${plan.total_budget:,.2f}", 0, 1)
    pdf.cell(0, 8, f"Timeline Summary: {plan.timeline_summary}", 0, 1)
    
    pdf.ln(5)
    
    # Detailed Plan Table (Shopping/Item List)
    pdf.set_fill_color(200, 220, 255)
    pdf.set_font("Arial", "B", 10)
    
    # Define Column Widths
    col_widths = [40, 30, 25, 95] 
    line_height = 6 

    # Table Header
    pdf.cell(col_widths[0], 7, "Item", 1, 0, "L", 1)
    pdf.cell(col_widths[1], 7, "Category", 1, 0, "L", 1)
    pdf.cell(col_widths[2], 7, "Est. Cost", 1, 0, "R", 1)
    pdf.cell(col_widths[3], 7, "Notes/Vendor Info", 1, 1, "L", 1)
    
    pdf.set_font("Arial", "", 10)
    total_estimated = 0
    
    # Table Rows (The original logic for multi-line cells is kept)
    for item in plan.items:
        x_start = pdf.get_x()
        y_start = pdf.get_y()
        
        # 2. Draw the Notes cell using multi_cell to determine the row height
        pdf.set_xy(x_start + col_widths[0] + col_widths[1] + col_widths[2], y_start)
        pdf.multi_cell(col_widths[3], line_height, item.notes, 0, "L") 
        
        # 3. Get the Y position *after* the multi_cell is drawn
        y_end = pdf.get_y()
        row_height = y_end - y_start
        
        # 4. Draw content and borders for ALL FOUR columns based on final row height
        
        # Draw Item Column
        pdf.set_xy(x_start, y_start)
        pdf.cell(col_widths[0], row_height, "", 1, 0, "L") 
        pdf.set_xy(x_start, y_start)
        pdf.multi_cell(col_widths[0], line_height, item.name, 0, "L") 
        
        # Draw Category Column
        pdf.set_xy(x_start + col_widths[0], y_start)
        pdf.cell(col_widths[1], row_height, "", 1, 0, "L") 
        pdf.set_xy(x_start + col_widths[0], y_start)
        pdf.multi_cell(col_widths[1], line_height, item.category, 0, "L") 
        
        # Draw Est. Cost Column
        pdf.set_xy(x_start + col_widths[0] + col_widths[1], y_start)
        pdf.cell(col_widths[2], row_height, "", 1, 0, "L") 
        pdf.set_xy(x_start + col_widths[0] + col_widths[1], y_start)
        pdf.multi_cell(col_widths[2], line_height, f"${item.estimated_cost_usd:,.2f}", 0, "R") 
        
        # Draw Notes Border (Content already drawn in Step 2)
        pdf.set_xy(x_start + col_widths[0] + col_widths[1] + col_widths[2], y_start)
        pdf.cell(col_widths[3], row_height, '', 1, 1, "L")
        
        # 5. Set the cursor to the starting X and ending Y for the next row
        pdf.set_xy(x_start, y_end)
        
        total_estimated += item.estimated_cost_usd
        
    pdf.ln(2)
    pdf.set_font("Arial", "B", 10)
    
    # --- FOOTER (Summary Total) ---
    merged_label_width = col_widths[0] + col_widths[1]

    pdf.cell(merged_label_width, 7, "TOTAL ESTIMATED SPEND", 1, 0, "R", 1)
    pdf.cell(col_widths[2], 7, f"${total_estimated:,.2f}", 1, 0, "R", 1)
    pdf.cell(col_widths[3], 7, f"Budget Balance: ${plan.total_budget - total_estimated:,.2f}", 1, 1, "L", 1)
    
    # Output to BytesIO object for Streamlit download
    pdf_output = pdf.output(dest='S').encode('latin1')
    return BytesIO(pdf_output)

# ==============================================================================
# 5. Main Streamlit App Execution Block
# ==============================================================================

def main():
    st.title("✨ AI Event Planner & Budget Assistant")
    st.markdown("Use the power of Gemini to generate a structured, budget-conscious event plan and PDF shopping list.")

    # --- Sidebar for API Key ---
    with st.sidebar:
        st.header("🔑 Gemini API Key")
        api_key = st.text_input(
            "Enter your Google AI Studio API Key:", 
            type="password",
            value=os.environ.get("GEMINI_API_KEY", "") # Uses environment variable if set
        )
        st.markdown("[Get your API Key here](https://aistudio.google.com/app/apikey)")
    
    if not api_key:
        st.warning("Please enter your Gemini API Key in the sidebar to proceed.")
        st.stop()

    # --- Main Input Form ---
    st.header("📝 Event Details")
    
    col1, col2 = st.columns(2)
    with col1:
        event_type = st.text_input("What type of event is this?", "Wedding")
        event_budget = st.number_input("Total event budget ($)", min_value=100.0, value=10000.0, step=100.0)
    
    with col2:
        event_theme = st.text_input("Desired theme", "Rustic Bohemian")
        num_people = st.number_input("Estimated number of guests", min_value=1, value=150, step=1)
    
    # Use a unique key for the button to reset the cache if inputs change drastically
    if st.button("Generate Event Plan", key="generate_plan_btn"):
        
        if not event_type or not event_theme or event_budget < 100 or num_people < 1:
            st.error("Please ensure all fields are filled out correctly.")
            st.stop()
            
        try:
            # 1. Execute the LLM Planning
            event_plan_data = generate_plan_with_llm(
                api_key, event_type, event_budget, num_people, event_theme
            )

            st.success("✅ Event plan successfully generated!")
            
            # --- Display Output ---
            st.header(f"Plan for: {event_plan_data.theme}")
            st.markdown(f"**Timeline:** {event_plan_data.timeline_summary}")
            st.metric(label="Total Budget", value=f"${event_plan_data.total_budget:,.2f}")
            
            # 2. Convert Pydantic list of items to a displayable DataFrame
            items_df = [item.dict() for item in event_plan_data.items]
            
            total_estimated = sum(item['estimated_cost_usd'] for item in items_df)
            
            st.subheader("Budget Breakdown & Shopping List")
            st.dataframe(items_df, use_container_width=True)
            
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.metric(label="Total Estimated Spend", value=f"${total_estimated:,.2f}")
            with col_b2:
                st.metric(label="Remaining Contingency/Buffer", value=f"${event_plan_data.total_budget - total_estimated:,.2f}")

            # 3. Generate PDF and provide download button
            pdf_buffer = generate_pdf(event_plan_data)
            
            clean_type = event_type.replace(' ', '_').replace('/', '_').replace('-', '_')
            clean_theme = event_theme.replace(' ', '_').replace('/', '_').replace('-', '_')
            filename = f"{clean_type}_{clean_theme}_Plan.pdf"

            st.download_button(
                label="Download PDF Plan & Shopping List",
                data=pdf_buffer,
                file_name=filename,
                mime="application/pdf"
            )

        except ConnectionError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()
