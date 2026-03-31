import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    from typing import List

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


@app.cell
def _():
    from langchain_openai import ChatOpenAI
    from dotenv import load_dotenv
    import os

    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    def get_llm():
        return ChatOpenAI(openai_api_key=openai_api_key,
                     model_name="gpt-5-nano")

    return (get_llm,)


@app.cell
def _():
    import json 

    def to_obj(s):
        try:
            return json.loads(s)
        except Exception:
            return {}

    return (to_obj,)


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
    Now that you have understood all the above, select the correct reserach assistant for the following question.
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


@app.cell
def _():
    NUM_SEARCH_QUERIES = 2
    NUM_SEARCH_RESULTS_PER_QUERY = 3
    RESULT_TEXT_MAX_CHARACTERS = 10000

    question = 'What can I see and do in the Spanish town of Astorga?'
    return (
        NUM_SEARCH_QUERIES,
        NUM_SEARCH_RESULTS_PER_QUERY,
        RESULT_TEXT_MAX_CHARACTERS,
        question,
    )


@app.cell
def _(get_llm):
    llm = get_llm()
    return (llm,)


@app.cell
def _(ASSISTANT_SELECTION_PROMPT_TEMPLATE, llm, mo, question, to_obj):
    assistant_selection_prompt = ASSISTANT_SELECTION_PROMPT_TEMPLATE.format(user_question=question)
    assistant_instructions = llm.invoke(assistant_selection_prompt)
    assistant_instructions_dict = to_obj(assistant_instructions.content)
    mo.Html(f"{assistant_instructions_dict}")
    return (assistant_instructions_dict,)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(
    NUM_SEARCH_QUERIES,
    WEB_SEARCH_PROMPT_TEMPLATE,
    assistant_instructions_dict,
    llm,
    mo,
    to_obj,
):
    web_search_prompt = WEB_SEARCH_PROMPT_TEMPLATE.format(
        assistant_instructions=assistant_instructions_dict[
            'assistant_instructions'],
        num_search_queries=NUM_SEARCH_QUERIES,
        user_question=assistant_instructions_dict[
            'user_question'])
    web_search_queries = llm.invoke(web_search_prompt)
    web_search_queries_list = to_obj(web_search_queries.content.replace('\n', ''))
    mo.Html(f"{web_search_queries_list}")
    return (web_search_queries_list,)


@app.cell
def _(NUM_SEARCH_RESULTS_PER_QUERY, web_search, web_search_queries_list):
    searches_and_result_urls = [{
            'result_urls': web_search(
        web_query=wq['search_query'],
        num_results=NUM_SEARCH_RESULTS_PER_QUERY),
            'search_query': wq['search_query']}
        for wq in web_search_queries_list]
    return (searches_and_result_urls,)


@app.cell
def _(mo, searches_and_result_urls):
    search_query_and_result_url_list = []
    for qr in searches_and_result_urls:
        search_query_and_result_url_list.extend([{
            'search_query': qr['search_query'],
            'result_url': r}
                for r in qr['result_urls']])

    mo.Html(f"{search_query_and_result_url_list}")
    return (search_query_and_result_url_list,)


@app.cell
def _(
    RESULT_TEXT_MAX_CHARACTERS,
    search_query_and_result_url_list,
    web_scrape,
):
    result_text_list = [{
        'result_text': web_scrape(
            url=re['result_url'])[:RESULT_TEXT_MAX_CHARACTERS],
        'result_url': re['result_url'],
        'search_query': re['search_query']}
                        for re in search_query_and_result_url_list]
    return (result_text_list,)


@app.cell
def _(SUMMARY_PROMPT_TEMPLATE, llm, result_text_list):
    result_text_summary_list = []
    for rt in result_text_list:
        summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(
            search_result_text=rt['result_text'],
            search_query=rt['search_query'])
        text_summary = llm.invoke(summary_prompt)
        result_text_summary_list.append(
            {'text_summary': text_summary,
             'result_url': rt['result_url'],
             'search_query': rt['search_query']})
    return (result_text_summary_list,)


@app.cell
def _(result_text_summary_list):
    stringified_summary_list = [
        f'Source URL: {sr["result_url"]}\nSummary: {sr["text_summary"]}'
            for sr in result_text_summary_list]
    return (stringified_summary_list,)


@app.cell
def _(stringified_summary_list):
    appended_result_summaries = '\n'.join(stringified_summary_list)
    return (appended_result_summaries,)


@app.cell
def _(
    RESEARCH_REPORT_PROMPT_TEMPLATE,
    appended_result_summaries,
    llm,
    mo,
    question,
):
    research_report_prompt = RESEARCH_REPORT_PROMPT_TEMPLATE.format(
        research_summary=appended_result_summaries,
        user_question=question
    )

    research_report = llm.invoke(research_report_prompt)

    # print(f'strigified_summary_list={stringified_summary_list}')
    # print(f'merged_result_summaries={appended_result_summaries}')
    mo.Html(f'research_report={research_report}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reimplementing the research summary engine in LCEL
    """)
    return


@app.cell
def _(ASSISTANT_SELECTION_PROMPT_TEMPLATE, get_llm, to_obj):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    assistant_instructions_chain = (
        {'user_question': RunnablePassthrough()}
        | ASSISTANT_SELECTION_PROMPT_TEMPLATE
        | get_llm() | StrOutputParser() | to_obj
    )
    return StrOutputParser, assistant_instructions_chain


@app.cell
def _(
    NUM_SEARCH_QUERIES,
    StrOutputParser,
    WEB_SEARCH_PROMPT_TEMPLATE,
    get_llm,
    to_obj,
):
    from langchain_core.runnables import RunnableLambda

    web_searches_chain = (
        RunnableLambda(lambda x:
            {
                'assistant_instructions': x['assistant_instructions'],
                'num_search_queries': NUM_SEARCH_QUERIES,
                'user_question': x['user_question']
            }
        )
        | WEB_SEARCH_PROMPT_TEMPLATE
        | get_llm() | StrOutputParser() | to_obj
    )
    return RunnableLambda, web_searches_chain


@app.cell
def _(NUM_SEARCH_RESULTS_PER_QUERY, RunnableLambda, web_search):
    search_result_urls_chain = (
        RunnableLambda(lambda x:
            [
                {
                'result_url': url,
                'search_query': x['search_query'],
                'user_question': x['user_question']
                }
                for url in web_search(
                    web_query=x['search_query'],
                    num_results=NUM_SEARCH_RESULTS_PER_QUERY)
            ]
        )
    )
    return (search_result_urls_chain,)


@app.cell
def _(
    RESULT_TEXT_MAX_CHARACTERS,
    RunnableLambda,
    SUMMARY_PROMPT_TEMPLATE,
    StrOutputParser,
    get_llm,
    web_scrape,
):
    from langchain_core.runnables import RunnableParallel

    search_result_text_and_summary_chain = (
        RunnableLambda(lambda x:
            {
                'search_result_text':
                    web_scrape(url=x['result_url'])[
                        :RESULT_TEXT_MAX_CHARACTERS],
                    'result_url': x['result_url'],
                    'search_query': x['search_query'],
                    'user_question': x['user_question']
            }
        )
        | RunnableParallel (
            {
                'text_summary': SUMMARY_PROMPT_TEMPLATE
                    | get_llm() | StrOutputParser(),
                'result_url': lambda x: x['result_url'],
                'user_question': lambda x: x['user_question']
            }
        )
        | RunnableLambda(lambda x:
            {
                'summary':
                    f"Source Url: {x['result_url']}\nSummary:{x['text_summary']}",
                'user_question': x['user_question']
            }
        )
    )
    return (search_result_text_and_summary_chain,)


@app.cell
def _(
    RunnableLambda,
    search_result_text_and_summary_chain,
    search_result_urls_chain,
):
    search_and_summarization_chain = (
        search_result_urls_chain
        | search_result_text_and_summary_chain.map() # parallelize for each url
        | RunnableLambda(lambda x:
            {
                'summary': '\n'.join([i['summary'] for i in x]),
                'user_question': x[0]['user_question'] if len(x) > 0 else ''
            })
    )
    return (search_and_summarization_chain,)


@app.cell
def _(
    RESEARCH_REPORT_PROMPT_TEMPLATE,
    RunnableLambda,
    StrOutputParser,
    assistant_instructions_chain,
    get_llm,
    search_and_summarization_chain,
    web_searches_chain,
):
    web_research_chain = (
        assistant_instructions_chain
        | web_searches_chain
        | search_and_summarization_chain.map() # parallelize for each web search
        | RunnableLambda(lambda x:
            {
            'research_summary': '\n\n'.join([i['summary'] for i in x]),
            'user_question': x[0]['user_question'] if len(x) > 0 else ''
            })
        | RESEARCH_REPORT_PROMPT_TEMPLATE | get_llm() | StrOutputParser()
    )
    return (web_research_chain,)


@app.cell
def _(mo, question, web_research_chain):
    web_research_report = web_research_chain.invoke(question)
    mo.Html(f"{web_research_report}")
    return


if __name__ == "__main__":
    app.run()
