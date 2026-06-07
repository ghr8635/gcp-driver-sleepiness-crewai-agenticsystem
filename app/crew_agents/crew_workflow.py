from crewai import Agent, Task, Crew, Process

from app.crew_agents.tools import run_tool_pipeline


def build_driver_sleepiness_crew() -> Crew:
    fatigue_agent = Agent(
        role="Fatigue Analysis Agent",
        goal="Analyze driver fatigue signals and support safe intervention decisions.",
        backstory=(
            "You are responsible for interpreting driver-state signals such as "
            "blink rate, yawning rate, PERCLOS, lane behavior, and steering behavior."
        ),
        verbose=True,
        allow_delegation=False,
    )

    rag_agent = Agent(
        role="RAG Retrieval Agent",
        goal="Retrieve relevant intervention knowledge from the FAISS safety memory.",
        backstory=(
            "You support the system by selecting useful safety context from the "
            "persistent intervention knowledge base."
        ),
        verbose=True,
        allow_delegation=False,
    )

    intervention_agent = Agent(
        role="Intervention Decision Agent",
        goal="Generate a safe in-cabin intervention for the detected fatigue risk.",
        backstory=(
            "You decide the final in-cabin alert action using fatigue score, risk level, "
            "retrieved safety context, and Gemini output."
        ),
        verbose=True,
        allow_delegation=False,
    )

    validation_agent = Agent(
        role="Safety Validation Agent",
        goal="Validate that the intervention is safe, structured, and suitable for the risk level.",
        backstory=(
            "You ensure that fan level, music state, vibration state, and reason are valid "
            "before the API returns a response."
        ),
        verbose=True,
        allow_delegation=False,
    )

    memory_agent = Agent(
        role="Memory Update Agent",
        goal="Update the FAISS vector memory when a generated intervention is semantically novel.",
        backstory=(
            "You manage adaptive memory by deciding whether new intervention knowledge "
            "should be stored in the persistent FAISS vector database."
        ),
        verbose=True,
        allow_delegation=False,
    )

    logging_agent = Agent(
        role="Logging Agent",
        goal="Ensure fatigue features and intervention decisions are logged for traceability.",
        backstory=(
            "You support observability by ensuring the prediction workflow is recorded "
            "in BigQuery for later analysis and monitoring."
        ),
        verbose=True,
        allow_delegation=False,
    )

    fatigue_task = Task(
        description="Analyze driver-state features and determine fatigue risk using the existing fatigue analysis logic.",
        expected_output="Fatigue score and risk level are calculated.",
        agent=fatigue_agent,
    )

    rag_task = Task(
        description="Retrieve relevant intervention knowledge from the persistent FAISS vector database.",
        expected_output="Relevant safety context and retrieval score are available.",
        agent=rag_agent,
    )

    intervention_task = Task(
        description="Generate an in-cabin intervention decision using retrieved context and fatigue risk.",
        expected_output="Fan level, music state, vibration state, and reason are generated.",
        agent=intervention_agent,
    )

    validation_task = Task(
        description="Validate the intervention format and enforce safe fallback rules if needed.",
        expected_output="Validated intervention decision is safe and structured.",
        agent=validation_agent,
    )

    memory_task = Task(
        description="Check whether the generated intervention is semantically novel and update vector memory if needed.",
        expected_output="Vector memory update status is available.",
        agent=memory_agent,
    )

    logging_task = Task(
        description="Log fatigue features and intervention decision to BigQuery for traceability.",
        expected_output="Prediction workflow is logged successfully.",
        agent=logging_agent,
    )

    crew = Crew(
        agents=[
            fatigue_agent,
            rag_agent,
            intervention_agent,
            validation_agent,
            memory_agent,
            logging_agent,
        ],
        tasks=[
            fatigue_task,
            rag_task,
            intervention_task,
            validation_task,
            memory_task,
            logging_task,
        ],
        process=Process.sequential,
        verbose=True,
    )

    return crew


def run_crewai_prediction(driver_state: dict) -> dict:
    """
    CrewAI workflow entry point.

    The CrewAI crew defines the multi-agent structure.
    The stable production prediction still runs through the existing Python tool pipeline.
    """

    crew = build_driver_sleepiness_crew()

    result = run_tool_pipeline(driver_state)

    result["workflow_version"] = "crew_v1_step_3"
    result["orchestration"] = "crewai_multi_agent_workflow"
    result["crew_metadata"] = {
        "crew_created": True,
        "agents_count": len(crew.agents),
        "tasks_count": len(crew.tasks),
        "process": "sequential",
    }
    result["agents_used"] = [
        "Fatigue Analysis Agent",
        "RAG Retrieval Agent",
        "Intervention Decision Agent",
        "Safety Validation Agent",
        "Memory Update Agent",
        "Logging Agent",
    ]

    return result