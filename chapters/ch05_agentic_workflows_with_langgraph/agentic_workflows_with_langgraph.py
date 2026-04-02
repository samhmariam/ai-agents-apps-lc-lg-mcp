import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step-by-step transformation process
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### STEP 1: DEFINE THE STATE
    """)
    return


@app.cell
def _():
    from langchain_openai import ChatOpenAI
    from typing import List, Dict, Any, TypedDict, Optional
    from dotenv import load_dotenv
    import os
    import json

    return (
        Any,
        ChatOpenAI,
        Dict,
        List,
        Optional,
        TypedDict,
        json,
        load_dotenv,
        os,
    )


@app.cell
def _(load_dotenv, os):
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return (openai_api_key,)


@app.cell
def _(ChatOpenAI, openai_api_key):
    def get_llm():
        return ChatOpenAI(openai_api_key=openai_api_key,
                     model_name="gpt-5-nano")

    return (get_llm,)


@app.cell
def _(Optional, TypedDict):
    # Define typed dictionaries for state handling
    class AssistantInfo(TypedDict):
        assistant_type: str
        assistant_instructions: str
        user_question: str

    class SearchQuery(TypedDict):
        search_query: str
        user_question: str

    class SearchResult(TypedDict):
        result_url: str
        search_query: str
        user_question: str
        is_fallback: Optional[bool]

    class SearchSummary(TypedDict):
        summary: str
        result_url: str
        user_question: str
        is_fallback: Optional[bool]

    class ResearchReport(TypedDict):
        report: str

    return AssistantInfo, SearchQuery, SearchResult, SearchSummary


@app.cell
def _(
    Any,
    AssistantInfo,
    Dict,
    List,
    Optional,
    SearchQuery,
    SearchResult,
    SearchSummary,
    TypedDict,
):
    # Graph state
    class ResearchState(TypedDict):
        user_question: str
        assistant_info: Optional[AssistantInfo]
        search_queries: Optional[List[SearchQuery]]
        search_results: Optional[List[SearchResult]]
        search_summaries: Optional[List[SearchSummary]]
        research_summary: Optional[str]
        final_report: Optional[str]
        used_fallback_search: Optional[bool]
        relevance_evaluation: Optional[Dict[str, Any]]
        should_regenerate_queries: Optional[bool]
        iteration_count: Optional[int]

    return (ResearchState,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Prompts
    """)
    return


@app.cell
def _():
    from langchain_core.prompts import PromptTemplate

    # web search and summarization prompts adapted from: https://github.com/langchain-ai/langchain/blob/master/templates/research-assistant/research_assistant/search/web.py 

    # ASSISTANT SELECTION
    ASSISTANT_SELECTION_INSTRUCTIONS = """
    You are skilled at assigning a research question to the correct research assistant. 
    There are various research assistants available, each specialized in an area of expertise. 
    Each assistant is identified by a specific type. Each assistant has specific instructions to undertake the research.

    How to select the correct assistant: you must select the relevant assistant depending on the topic of the question, which should match the area of expertise of the assistant.

    ------
    Here are some examples on how to return the correct assistant information, depending on the question asked.

    Examples:
    Question: "Should I invest in Apple stocks?"
    Response: 
    {{
        "assistant_type": "Financial analyst assistant",
        "assistant_instructions": "You are a seasoned finance analyst AI assistant. Your primary goal is to compose comprehensive, astute, impartial, and methodically arranged financial reports based on provided data and trends.",
        "user_question": {user_question}
    }}
    Question: "what are the most interesting sites in Tel Aviv?"
    Response: 
    {{
        "assistant_type": "Tour guide assistant",
        "assistant_instructions": "You are a world-travelled AI tour guide assistant. Your main purpose is to draft engaging, insightful, unbiased, and well-structured travel reports on given locations, including history, attractions, and cultural insights.",
        "user_question": "{user_question}"
    }}

    Question: "Is Messi a good soccer player?"
    Response: 
    {{
        "assistant_type": "Sport expert assistant",
        "assistant_instructions": "You are an experienced AI sport assistant. Your main purpose is to draft engaging, insightful, unbiased, and well-structured sport reports on given sport personalities, or sport events, including factual details, statistics and insights.",
        "user_question": "{user_question}"
    }}

    ------
    Now that you have understood all the above, select the correct research assistant for the following question.
    Question: {user_question}
    Response:

    """ 

    ASSISTANT_SELECTION_PROMPT_TEMPLATE = PromptTemplate.from_template( 
        template=ASSISTANT_SELECTION_INSTRUCTIONS
    )

    # WEB SEARCH
    WEB_SEARCH_INSTRUCTIONS = """
    {assistant_instructions}

    Write {num_search_queries} web search queries to gather as much information as possible 
    on the following question: {user_question}. Your objective is to write a report based on the information you find.
    You must respond with a list of queries such as query1, query2, query3 in the following format: 
    [
        {{"search_query": "query1", "user_question": "{user_question}" }},
        {{"search_query": "query2", "user_question": "{user_question}" }},
        {{"search_query": "query3", "user_question": "{user_question}" }}
    ]
    """

    WEB_SEARCH_PROMPT_TEMPLATE = PromptTemplate.from_template(
        template=WEB_SEARCH_INSTRUCTIONS
    )

    # INDIVIDUAL SEARCH SUMMARY
    SUMMARY_INSTRUCTIONS = """
    Read the following text:
    Text: {search_result_text} 

    -----------

    Using the above text, answer in short the following question.
    Question: {search_query}
 
    -----------
    If you cannot answer the question above using the text provided above, then just summarize the text. 
    Include all factual information, numbers, stats etc if available.
    """

    SUMMARY_PROMPT_TEMPLATE = PromptTemplate.from_template(
        template=SUMMARY_INSTRUCTIONS
    )

    # RESEARCH REPORT
    # Research Report prompts adapted from https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/master/prompts.py
    RESEARCH_REPORT_INSTRUCTIONS = """
    You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text.

    Information: 
    --------
    {research_summary}
    --------

    Using the above information, answer the following question or topic: "{user_question}" in a detailed report -- \
    The report should focus on the answer to the question, should be well structured, informative, \
    in depth, with facts and numbers if available and a minimum of 1,200 words.

    You should strive to write the report as long as you can using all relevant and necessary information provided.
    You must write the report with markdown syntax.
    You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
    Write all used source urls at the end of the report, and make sure to not add duplicated sources, but only one reference for each.
    You must write the report in apa format.
    Please do your best, this is very important to my career.""" 

    RESEARCH_REPORT_PROMPT_TEMPLATE = PromptTemplate.from_template(
        template=RESEARCH_REPORT_INSTRUCTIONS
    )
    return (
        ASSISTANT_SELECTION_PROMPT_TEMPLATE,
        RESEARCH_REPORT_PROMPT_TEMPLATE,
        SUMMARY_PROMPT_TEMPLATE,
        WEB_SEARCH_PROMPT_TEMPLATE,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### STEP 2: CONVERT COMPONENTS TO NODE FUNCTIONS
    """)
    return


@app.cell
def _():
    from langchain_core.output_parsers import StrOutputParser

    return


@app.cell
def _(ASSISTANT_SELECTION_PROMPT_TEMPLATE, Any, Dict, get_llm, json):
    def select_assistant(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select the appropriate research assistant based on the user question.
        """
        user_question = state["user_question"]
    
        # Format the prompt with the user question
        prompt = ASSISTANT_SELECTION_PROMPT_TEMPLATE.format(user_question=user_question)
    
        # Get the LLM response
        llm = get_llm()
        response = llm.invoke(prompt)
        response_text = response.content
    
        # Parse the response to get the assistant info
        try:
            # Extract the JSON part from the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            json_str = response_text[json_start:json_end]
        
            # Parse the JSON
            assistant_info = json.loads(json_str)
        
            # Return the updated state
            return {"assistant_info": assistant_info}
        except Exception as e:
            # Fallback to a default assistant if parsing fails
            default_assistant = {
                "assistant_type": "General research assistant",
                "assistant_instructions": "You are a general research AI assistant. Your main purpose is to draft comprehensive, informative, unbiased, and well-structured reports on given topics.",
                "user_question": user_question
            }
            return {"assistant_info": default_assistant}

    return (select_assistant,)


@app.cell
def _():
    NUM_SEARCH_QUERIES = 3
    NUM_SEARCH_RESULTS_PER_QUERY = 3
    RESULT_TEXT_MAX_CHARACTERS = 10000
    return (
        NUM_SEARCH_QUERIES,
        NUM_SEARCH_RESULTS_PER_QUERY,
        RESULT_TEXT_MAX_CHARACTERS,
    )


@app.cell
def _(
    Any,
    Dict,
    NUM_SEARCH_QUERIES,
    WEB_SEARCH_PROMPT_TEMPLATE,
    get_llm,
    json,
):
    def generate_search_queries(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate search queries based on the assistant instructions and user question.
        Uses different strategies based on iteration count to ensure variety.
        """
        assistant_info = state["assistant_info"]
        user_question = state["user_question"]
        assistant_instructions = assistant_info["assistant_instructions"]
    
        # Get the current iteration count
        iteration_count = state.get("iteration_count", 0)
    
        # Check if this is a regeneration (we already have search queries and relevance evaluation)
        previous_queries = state.get("search_queries", [])
        relevance_evaluation = state.get("relevance_evaluation", None)
    
        # Format the prompt based on iteration count
        if iteration_count == 0:
            # First-time query generation
            print("Generating initial search queries...")
            prompt = WEB_SEARCH_PROMPT_TEMPLATE.format(
                assistant_instructions=assistant_instructions,
                user_question=user_question,
                num_search_queries=NUM_SEARCH_QUERIES
            )
        elif iteration_count == 1:
            # Second iteration - more specific queries
            print("First regeneration: Creating more specific queries...")
            previous_query_list = ", ".join([q["search_query"] for q in previous_queries])
            relevance_percentage = relevance_evaluation.get("relevance_percentage", 0) if relevance_evaluation else 0
            relevance_explanation = relevance_evaluation.get("explanation", "No explanation provided") if relevance_evaluation else ""
        
            prompt = f"""
            {assistant_instructions}

            You are generating new search queries because the previous queries did not yield sufficiently relevant results.
        
            Original question: {user_question}
        
            Previous search queries: {previous_query_list}
        
            Relevance evaluation: {relevance_percentage}% relevant
            Explanation: {relevance_explanation}
        
            Please generate {NUM_SEARCH_QUERIES} NEW and DIFFERENT web search queries that are MORE SPECIFIC and TARGETED 
            to gather relevant information on the original question. 
        
            IMPORTANT: DO NOT repeat or rephrase the previous queries. Create completely different approaches to finding information.
        
            You must respond with a list of queries in the following format:
            [
                {{"search_query": "query1", "user_question": "{user_question}" }},
                {{"search_query": "query2", "user_question": "{user_question}" }},
                {{"search_query": "query3", "user_question": "{user_question}" }}
            ]
            """
        else:
            # Third or later iteration - completely different approach
            print(f"Iteration {iteration_count}: Using alternative search strategies...")
            all_previous_queries = ", ".join([q["search_query"] for q in previous_queries])
        
            prompt = f"""
            {assistant_instructions}

            You are generating search queries for the FINAL attempt to find relevant information.
        
            Original question: {user_question}
        
            All previous search queries that DID NOT yield relevant results: {all_previous_queries}
        
            For this final attempt, take a completely different angle. Consider:
            1. Breaking down the question into smaller, more focused sub-questions
            2. Using technical or specialized terms related to the topic
            3. Searching for expert opinions or academic perspectives
            4. Looking for case studies or specific examples
            5. Exploring historical context or background information
        
            CRITICAL INSTRUCTIONS:
            1. DO NOT repeat or rephrase ANY previous queries listed above
            2. Generate queries that are COMPLETELY DIFFERENT from all previous attempts
        
            Please generate {NUM_SEARCH_QUERIES} COMPLETELY NEW search queries following the strategy above.
        
            You must respond with a list of queries in the following format:
            [
                {{"search_query": "query1", "user_question": "{user_question}" }},
                {{"search_query": "query2", "user_question": "{user_question}" }},
                {{"search_query": "query3", "user_question": "{user_question}" }}
            ]
            """
    
        # Get the LLM response
        llm = get_llm()
        response = llm.invoke(prompt)
        response_text = response.content
    
        # Parse the response to get the search queries
        try:
            # Extract the JSON array from the response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            json_str = response_text[json_start:json_end]
        
            # Parse the JSON
            search_queries = json.loads(json_str)
        
            print(f"Generated {len(search_queries)} search queries")
            for i, query in enumerate(search_queries):
                print(f"  Query {i+1}: {query['search_query']}")
        
            # Return the updated state
            return {
                "search_queries": search_queries,
                # Reset the relevance evaluation and regeneration flag when generating new queries
                "relevance_evaluation": None,
                "should_regenerate_queries": None
            }
        except Exception as e:
            print(f"Error parsing search queries: {str(e)}")
            # Fallback to a default search query if parsing fails
            default_queries = [
                {"search_query": f"{user_question} iteration {iteration_count + 1}", "user_question": user_question}
            ]
            print(f"Using default query: {default_queries[0]['search_query']}")
            return {
                "search_queries": default_queries,
                "relevance_evaluation": None,
                "should_regenerate_queries": None
            }

    return (generate_search_queries,)


@app.cell
def _(Any, Dict, NUM_SEARCH_RESULTS_PER_QUERY, web_search):
    def perform_web_searches(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform web searches based on the generated search queries.
        """
        search_queries = state["search_queries"]
        search_results = []
        fallback_used = False
    
        print(f"Performing web searches for {len(search_queries)} queries...")
    
        # For each search query, get the search results
        for query_obj in search_queries:
            search_query = query_obj["search_query"]
            user_question = query_obj["user_question"]
        
            try:
                # Get the search results
                print(f"Searching for: {search_query}")
                urls = web_search(web_query=search_query, num_results=NUM_SEARCH_RESULTS_PER_QUERY)
            
                # Check if these are likely fallback results (Wikipedia URLs)
                if any("wikipedia.org" in url for url in urls[:2]):
                    print(f"Fallback search was used for query: {search_query}")
                    fallback_used = True
                    is_fallback = True
                else:
                    is_fallback = False
            
                # Add the results to the list
                for url in urls:
                    search_results.append({
                        "result_url": url,
                        "search_query": search_query,
                        "user_question": user_question,
                        "is_fallback": is_fallback
                    })
                
                print(f"Found {len(urls)} results for query: {search_query}")
            except Exception as e:
                print(f"Error searching for '{search_query}': {str(e)}")
                # Continue with other queries even if one fails
                continue
    
        # If we have no search results at all, add a fallback result
        if not search_results:
            print("No search results found. Using general fallback information.")
            fallback_url = "https://en.wikipedia.org/wiki/Main_Page"
            search_results.append({
                "result_url": fallback_url,
                "search_query": "general information",
                "user_question": state["user_question"],
                "is_fallback": True
            })
            fallback_used = True
    
        # Return the updated state with information about fallback usage
        return {
            "search_results": search_results,
            "used_fallback_search": fallback_used
        }

    return (perform_web_searches,)


@app.cell
def _(
    Any,
    Dict,
    RESULT_TEXT_MAX_CHARACTERS,
    SUMMARY_PROMPT_TEMPLATE,
    get_llm,
    web_scrape,
):
    def summarize_search_results(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize the search results.
        """
        search_results = state["search_results"]
        used_fallback_search = state.get("used_fallback_search", False)
        llm = get_llm()
        summaries = []
    
        print(f"Summarizing {len(search_results)} search results...")
    
        # For each search result, get the text and summarize it
        for result in search_results:
            result_url = result["result_url"]
            search_query = result["search_query"]
            user_question = result["user_question"]
            is_fallback = result.get("is_fallback", False)
        
            try:
                # Get the webpage content
                print(f"Scraping content from: {result_url}")
                search_result_text = web_scrape(url=result_url)[:RESULT_TEXT_MAX_CHARACTERS]
            
                # Skip if the content is an error message or too short
                if search_result_text.startswith("Failed to") or len(search_result_text) < 50:
                    print(f"Skipping {result_url} due to scraping issues or insufficient content")
                    continue
            
                # Format the prompt, with additional context for fallback results
                if is_fallback:
                    prompt = f"""
                    You are summarizing content from a fallback source that was used because the primary search engine was unavailable.
                
                    Read the following text:
                    Text: {search_result_text} 
                
                    -----------
                
                    Using the above text, answer in short the following question.
                    Question: {search_query}
                
                    -----------
                    If you cannot answer the question above using the text provided above, then just summarize the text. 
                    Include all factual information, numbers, stats etc if available.
                
                    Note that this is a fallback source, so it might not directly address the question.
                    """
                else:
                    prompt = SUMMARY_PROMPT_TEMPLATE.format(
                        search_result_text=search_result_text,
                        search_query=search_query
                    )
            
                # Get the summary
                summary_response = llm.invoke(prompt)
                text_summary = summary_response.content
            
                # Add a note about fallback sources
                if is_fallback:
                    source_note = "[Note: This information comes from a fallback source and may not directly address the question.]"
                    text_summary = f"{text_summary}\n{source_note}"
            
                # Create the summary object
                summary = {
                    "summary": f"Source Url: {result_url}\nSummary: {text_summary}",
                    "result_url": result_url,
                    "user_question": user_question,
                    "is_fallback": is_fallback
                }
            
                summaries.append(summary)
                print(f"Successfully summarized content from: {result_url}")
            except Exception as e:
                print(f"Error summarizing {result_url}: {str(e)}")
                # Skip this result if there's an error
                continue
    
        # Create the research summary
        if summaries:
            research_summary = "\n\n".join([s["summary"] for s in summaries])
            print(f"Created research summary with {len(summaries)} sources")
        
            # Add a note if fallback search was used
            if used_fallback_search:
                fallback_note = "\n\n[Note: Some or all of this information comes from fallback sources because the primary search engine was unavailable. The information may not be as directly relevant to your question as usual.]"
                research_summary += fallback_note
        else:
            research_summary = "No relevant information found. Please try different search queries."
            print("Warning: No summaries were generated from search results")
    
        # Return the updated state
        return {
            "search_summaries": summaries,
            "research_summary": research_summary,
            "used_fallback_search": used_fallback_search
        }

    return (summarize_search_results,)


@app.cell
def _(Any, Dict, get_llm, json):
    def evaluate_search_relevance(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the relevance of search summaries to the original question.
        If less than 50% of summaries are relevant, return to search query generation.
        """
        search_summaries = state.get("search_summaries", [])
        user_question = state["user_question"]
        research_summary = state.get("research_summary", "")
        used_fallback_search = state.get("used_fallback_search", False)
    
        print("Evaluating relevance of search summaries to the original question...")
    
        # If there are no summaries, we need to regenerate queries
        if not search_summaries or not research_summary:
            print("No search summaries found. Regenerating search queries...")
            return {"should_regenerate_queries": True}
    
        # Use the LLM to evaluate relevance
        llm = get_llm()
    
        # Create a prompt for the LLM to evaluate relevance
        evaluation_prompt = f"""
        You are an expert research evaluator. Your task is to evaluate the relevance of search results 
        to the original research question.
    
        Original research question: {user_question}
    
        Search result summaries:
        {research_summary}
    
        For each search result summary, determine if it is relevant to answering the original question.
        Then calculate what percentage of the search results are relevant.
    
        Return your evaluation as a JSON object with the following structure:
        {{
            "relevance_percentage": <percentage of relevant results as a number between 0 and 100>,
            "explanation": <brief explanation of your evaluation>,
            "relevant_count": <number of relevant summaries>,
            "total_count": <total number of summaries>
        }}
        """
    
        try:
            # Get the evaluation from the LLM
            evaluation_response = llm.invoke(evaluation_prompt)
            evaluation_text = evaluation_response.content
        
            # Extract the JSON from the response
            try:
                # Find JSON in the response
                json_start = evaluation_text.find('{')
                json_end = evaluation_text.rfind('}') + 1
                json_str = evaluation_text[json_start:json_end]
            
                # Parse the JSON
                evaluation = json.loads(json_str)
                relevance_percentage = evaluation.get("relevance_percentage", 0)
            
                # Determine if we should regenerate queries (less than 50% relevant)
                should_regenerate = relevance_percentage < 50
            
                if should_regenerate:
                    print(f"Only {relevance_percentage}% of search results are relevant. Regenerating search queries...")
                else:
                    print(f"{relevance_percentage}% of search results are relevant. Proceeding to write research report...")
            
                return {
                    "relevance_evaluation": evaluation,
                    "should_regenerate_queries": should_regenerate
                }
            except Exception as e:
                print(f"Error parsing relevance evaluation: {str(e)}")
                # If we can't parse the evaluation, assume we need to regenerate
                return {"should_regenerate_queries": True}
        except Exception as e:
            print(f"Error during relevance evaluation: {str(e)}")
            # If there's an error, assume we need to regenerate
            return {"should_regenerate_queries": True}

    return (evaluate_search_relevance,)


@app.cell
def _(Any, Dict, RESEARCH_REPORT_PROMPT_TEMPLATE, get_llm):
    def write_research_report(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Write a research report based on the summarized search results.
        """
        research_summary = state["research_summary"]
        user_question = state["user_question"]
    
        # Format the prompt
        prompt = RESEARCH_REPORT_PROMPT_TEMPLATE.format(
            research_summary=research_summary,
            user_question=user_question
        )
    
        # Get the LLM response
        llm = get_llm()
        response = llm.invoke(prompt)
        report = response.content
    
        # Return the updated state
        return {"final_report": report}

    return (write_research_report,)


@app.cell
def _(List):
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

    def web_search(
        web_query: str, 
        num_results: int) -> List[str]:
        return [r["link"] 
            for r in DuckDuckGoSearchAPIWrapper().results(
                web_query, num_results)]

    return (web_search,)


@app.cell
def _():
    import requests
    from bs4 import BeautifulSoup

    def web_scrape(url: str) -> str:
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            }
            response = requests.get(url, headers=headers, timeout=15)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                page_text = soup.get_text(separator=" ", strip=True)

                return page_text
            else:
                return f"Failed to retrieve the webpage: Status code {response.status_code}"
        except Exception as e:
            print(e)
            return f"Failed to retrieve the webpage: {e}"

    return (web_scrape,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### STEP 3: DEFINE THE GRAPH STRUCTURE
    """)
    return


@app.cell
def _():
    from langgraph.graph import StateGraph, END

    return END, StateGraph


@app.cell
def _(
    Any,
    Dict,
    END,
    ResearchState,
    StateGraph,
    evaluate_search_relevance,
    generate_search_queries,
    perform_web_searches,
    select_assistant,
    summarize_search_results,
    write_research_report,
):
    def create_research_graph() -> StateGraph:
        """
        Create the LangGraph research graph that coordinates the agents.
        """
        # Define the graph
        graph = StateGraph(ResearchState)
    
        # Add nodes to the graph
        graph.add_node("select_assistant", select_assistant)
        graph.add_node("generate_search_queries", generate_search_queries)
        graph.add_node("perform_web_searches", perform_web_searches)
        graph.add_node("summarize_search_results", summarize_search_results)
        graph.add_node("evaluate_search_relevance", evaluate_search_relevance)
        graph.add_node("write_research_report", write_research_report)
    
        # Define the conditional routing function for relevance evaluation
        def route_based_on_relevance(state: Dict[str, Any]) -> str:
            """
            Route to either generate new search queries or continue to report writing
            based on the relevance evaluation.
            """
            # Get the current iteration count
            iteration_count = state.get("iteration_count", 0)
        
            # Increment the iteration count
            new_iteration_count = iteration_count + 1
        
            # Update the state with the new iteration count
            state["iteration_count"] = new_iteration_count
        
            # Check if we've reached the maximum number of iterations (3)
            if new_iteration_count >= 3:
                print(f"Reached maximum iterations ({new_iteration_count}). Proceeding to write report with current results.")
                return "write_research_report"
        
            # Otherwise, check if we should regenerate queries
            if state.get("should_regenerate_queries", False):
                print(f"Iteration {new_iteration_count}: Regenerating search queries.")
                return "generate_search_queries"
            else:
                print(f"Iteration {new_iteration_count}: Search results are relevant. Proceeding to write report.")
                return "write_research_report"
    
        # Define the flow of the graph
        graph.add_edge("select_assistant", "generate_search_queries")
        graph.add_edge("generate_search_queries", "perform_web_searches")
        graph.add_edge("perform_web_searches", "summarize_search_results")
        graph.add_edge("summarize_search_results", "evaluate_search_relevance")
    
        # Add conditional routing based on relevance evaluation
        graph.add_conditional_edges(
            "evaluate_search_relevance",
            route_based_on_relevance,
            {
                "generate_search_queries": "generate_search_queries",
                "write_research_report": "write_research_report"
            }
        )
    
        graph.add_edge("write_research_report", END)
    
        # Set the entry point
        graph.set_entry_point("select_assistant")
    
        return graph

    return (create_research_graph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### STEP 4: COMPILE AND RUN THE GRAPH
    """)
    return


@app.cell
def _(create_research_graph):
    def run_research(question: str) -> str:
        """
        Run the research graph with a user question.
    
        Args:
            question: The user's research question
        
        Returns:
            The final research report
        """
        # Create the graph
        research_graph = create_research_graph()
    
        # Compile the graph
        app = research_graph.compile()
    
        # Initialize the state
        initial_state = {
            "user_question": question,
            "assistant_info": None,
            "search_queries": None,
            "search_results": None,
            "search_summaries": None,
            "research_summary": None,
            "final_report": None,
            "used_fallback_search": False,
            "relevance_evaluation": None,
            "should_regenerate_queries": None,
            "iteration_count": 0
        }
    
        # Run the graph
        result = app.invoke(initial_state)
    
        # Extract and return the final report
        return result["final_report"]

    # For testing purposes
    if __name__ == "__main__":
        # Example usage
        question = "What can you tell me about Astorga's roman spas"
        report = run_research(question)
        print(report)
    return


@app.cell
def _():
    # Visualize the DAG for the LangGraph research process
    try:
        from graphviz import Digraph
    except Exception as e:
        Digraph = None
        print("graphviz package is not available. A DOT representation will be generated as fallback.")

    def visualize_research_dag():
        """
        Build and visualize the research graph DAG.
        Returns a Graphviz Digraph object if available, otherwise a DOT string.
        """
        if Digraph is None:
            # Fallback DOT representation
            dot_fallback = """digraph ResearchGraph {
    rankdir=LR;
    node [shape=box, style=rounded];
    select_assistant -> generate_search_queries;
    generate_search_queries -> perform_web_searches;
    perform_web_searches -> summarize_search_results;
    summarize_search_results -> evaluate_search_relevance;
    evaluate_search_relevance -> generate_search_queries [label="regenerate_queries"];
    evaluate_search_relevance -> write_research_report [label="proceed_to_report"];
    write_research_report -> END;
    }"""
            return dot_fallback
        # Use Graphviz to render a DAG
        dag = Digraph('ResearchGraph', comment='LangGraph Research DAG')
        dag.attr(rankdir='LR')
        dag.attr('node', shape='box', style='rounded')
    
        # Define nodes (optional emotion/labels can be adjusted)
        dag.node('select_assistant', 'select_assistant')
        dag.node('generate_search_queries', 'generate_search_queries')
        dag.node('perform_web_searches', 'perform_web_searches')
        dag.node('summarize_search_results', 'summarize_search_results')
        dag.node('evaluate_search_relevance', 'evaluate_search_relevance')
        dag.node('write_research_report', 'write_research_report')
        dag.node('END', 'END')
    
        # Add edges to represent the DAG flow
        dag.edge('select_assistant', 'generate_search_queries')
        dag.edge('generate_search_queries', 'perform_web_searches')
        dag.edge('perform_web_searches', 'summarize_search_results')
        dag.edge('summarize_search_results', 'evaluate_search_relevance')
        dag.edge('evaluate_search_relevance', 'generate_search_queries', label='regenerate_queries')
        dag.edge('evaluate_search_relevance', 'write_research_report', label='proceed')
        dag.edge('write_research_report', 'END')
    
        return dag

    dag_visualization = visualize_research_dag()
    dag_visualization
    return


if __name__ == "__main__":
    app.run()
