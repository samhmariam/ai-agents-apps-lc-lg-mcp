import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import os
    print("Current working directory:", os.getcwd())
    return


@app.cell
def _():
    with open(".\data\Moby-Dick.txt", 'r', encoding='utf-8') as f:
        moby_dick_book = f.read()
    return (moby_dick_book,)


@app.cell
def _():
    from langchain_openai import ChatOpenAI
    from langchain_text_splitters import TokenTextSplitter
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda, RunnableParallel
    import getpass

    return (
        ChatOpenAI,
        PromptTemplate,
        RunnableLambda,
        RunnableParallel,
        StrOutputParser,
        TokenTextSplitter,
        getpass,
    )


@app.cell
def _(getpass):
    OPENAI_API_KEY = getpass.getpass('Enter your OPENAI_API_KEY')
    return (OPENAI_API_KEY,)


@app.cell
def _(ChatOpenAI, OPENAI_API_KEY):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,model_name="gpt-5-nano")
    return (llm,)


@app.cell
def _(RunnableLambda, TokenTextSplitter):
    # Split
    text_chunks_chain = (
        RunnableLambda(lambda x: 
            [
                {
                    'chunk': text_chunk, 
                }
                for text_chunk in 
                   TokenTextSplitter(chunk_size=3000, chunk_overlap=100).split_text(x)
            ]
        )
    )
    return (text_chunks_chain,)


@app.cell
def _(PromptTemplate, RunnableParallel, StrOutputParser, llm):
    # Map
    summarize_chunk_prompt_template = """
    Write a concise summary of the following text, and include the main details.
    Text: {chunk}
    """

    summarize_chunk_prompt = PromptTemplate.from_template(summarize_chunk_prompt_template)
    summarize_chunk_chain = summarize_chunk_prompt | llm

    summarize_map_chain = (
        RunnableParallel (
            {
                'summary': summarize_chunk_chain | StrOutputParser()        
            }
        )
    )
    return (summarize_map_chain,)


@app.cell
def _(PromptTemplate, RunnableLambda, StrOutputParser, llm):
    # Reduce
    summarize_summaries_prompt_template = """
    Write a coincise summary of the following text, which joins several summaries, and include the main details.
    Text: {summaries}
    """

    summarize_summaries_prompt = PromptTemplate.from_template(summarize_summaries_prompt_template)
    summarize_reduce_chain = (
        RunnableLambda(lambda x: 
            {
                'summaries': '\n'.join([i['summary'] for i in x]), 
            })
        | summarize_summaries_prompt 
        | llm 
        | StrOutputParser()
    )
    return (summarize_reduce_chain,)


@app.cell
def _(summarize_map_chain, summarize_reduce_chain, text_chunks_chain):
    map_reduce_chain = (
       text_chunks_chain
       | summarize_map_chain.map()
       | summarize_reduce_chain
    )     
    return (map_reduce_chain,)


@app.cell
def _(map_reduce_chain, moby_dick_book):
    summary = map_reduce_chain.invoke(moby_dick_book)
    return (summary,)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo, summary):
    mo.Html(f"{summary}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Summarizing across documents
    """)
    return


@app.cell
def _():
    from langchain_community.document_loaders import WikipediaLoader

    wikipedia_loader = WikipediaLoader(query="Paestum", load_max_docs=2)
    wikipedia_docs = wikipedia_loader.load()
    return (wikipedia_docs,)


@app.cell
def _():
    from langchain_community.document_loaders import Docx2txtLoader
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.document_loaders import TextLoader

    word_loader = Docx2txtLoader("./data\Paestum\Paestum-Britannica.docx")
    word_docs = word_loader.load()

    pdf_loader = PyPDFLoader("./data\Paestum\PaestumRevisited.pdf")
    pdf_docs = pdf_loader.load()

    txt_loader = TextLoader("./data\Paestum\Paestum-Encyclopedia.txt")
    txt_docs = txt_loader.load()
    return pdf_docs, txt_docs, word_docs


@app.cell
def _(pdf_docs, txt_docs, wikipedia_docs, word_docs):
    all_docs = wikipedia_docs + word_docs + pdf_docs + txt_docs
    return (all_docs,)


@app.cell
def _(PromptTemplate, llm):
    doc_summary_template = """Write a concise summary of the following text:
    {text}
    DOC SUMMARY:"""
    doc_summary_prompt = PromptTemplate.from_template(doc_summary_template)

    doc_summary_chain = doc_summary_prompt | llm
    return


@app.cell
def _(PromptTemplate, StrOutputParser, llm):
    refine_summary_template = """
    Your must produce a final summary from the current refined summary
    which has been generated so far and from the content of an additional document.
    This is the current refined summary generated so far: {current_refined_summary}
    This is the content of the additional document: {text}
    Only use the content of the additional document if it is useful, 
    otherwise return the current full summary as it is."""

    refine_summary_prompt = PromptTemplate.from_template(refine_summary_template)

    refine_chain = refine_summary_prompt | llm | StrOutputParser()
    return (refine_chain,)


@app.cell
def _(refine_chain):
    def refine_summary(docs):

        intermediate_steps = []
        current_refined_summary = ''
        for doc in docs:
            intermediate_step = \
               {"current_refined_summary": current_refined_summary, 
                "text": doc.page_content}
            intermediate_steps.append(intermediate_step)
        
            current_refined_summary = refine_chain.invoke(intermediate_step)
        
        return {"final_summary": current_refined_summary,
                "intermediate_steps": intermediate_steps}

    return (refine_summary,)


@app.cell
def _(all_docs, refine_summary):
    full_summary = refine_summary(all_docs)
    print(full_summary)
    return


if __name__ == "__main__":
    app.run()
