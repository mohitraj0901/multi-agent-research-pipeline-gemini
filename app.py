import streamlit as st

from workflows.graph_builder import create_workflow
from state.schemas import create_initial_state
from workflows.checkpointer import create_thread_id


st.set_page_config(
    page_title="Multi-Agent Research Pipeline",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Agent Research & Content Pipeline")
st.caption("LangGraph + Gemini + Tavily + Automated Fact Checker")

st.markdown(
    """
    Enter a research topic and let the multi-agent workflow:
    **Research → Create Content → Fact Check → Review → Final Output**
    """
)

task = st.text_area(
    "Research Topic",
    placeholder="Example: Explain how transformer architecture works in deep learning",
    height=120
)

col1, col2 = st.columns(2)

with col1:
    content_type = st.selectbox(
        "Content Type",
        ["blog post", "article", "report", "summary"]
    )

with col2:
    audience = st.text_input(
        "Target Audience",
        value="general audience"
    )

run_pipeline = st.button(
    "🚀 Run Research Pipeline",
    type="primary",
    use_container_width=True
)

if run_pipeline:
    if not task.strip():
        st.warning("Please enter a research topic.")
        st.stop()

    with st.spinner("Running multi-agent workflow... This may take a while."):
        try:
            initial_state = create_initial_state(
                task=task,
                context={
                    "content_type": content_type,
                    "audience": audience
                }
            )

            workflow = create_workflow(with_checkpointing=True)

            thread_id = create_thread_id(task)
            config = {"configurable": {"thread_id": thread_id}}

            progress = st.empty()

            step = 0

            for event in workflow.stream(initial_state, config):
                step += 1
                agent_name = list(event.keys())[0]

                progress.info(
                    f"Step {step}: {agent_name.replace('_', ' ').title()} completed"
                )

            final_state = workflow.get_state(config)
            state = final_state.values

            progress.success("✅ Workflow completed!")

            if state.get("final_output"):
                st.subheader("📝 Final Output")
                st.markdown(state["final_output"])

                metadata = state.get("metadata", {})

                st.subheader("📊 Pipeline Statistics")

                m1, m2, m3, m4 = st.columns(4)

                m1.metric(
                    "Word Count",
                    metadata.get("word_count", "N/A")
                )

                m2.metric(
                    "Quality Score",
                    f"{metadata.get('quality_score', 0):.2f}/1.0"
                )

                m3.metric(
                    "Fact Check Score",
                    f"{state.get('fact_check_score', 0):.2f}"
                )

                m4.metric(
                    "Iterations",
                    state.get("iteration_count", "N/A")
                )

                with st.expander("🔎 Fact Check Report"):
                    st.write(
                        state.get(
                            "fact_check_report",
                            "No fact-check report available."
                        )
                    )

            else:
                st.error("Workflow did not produce final output.")

                if state.get("errors"):
                    st.write("Errors:")
                    for error in state["errors"]:
                        st.write(f"- {error}")

        except Exception as e:
            st.error(f"Pipeline failed: {e}")