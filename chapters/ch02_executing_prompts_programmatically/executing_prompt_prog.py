import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### OpenAI API prompt examples
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from openai import OpenAI
    import getpass

    OPENAI_API_KEY = getpass.getpass('Enter your OPENAI_API_KEY') 
    return OPENAI_API_KEY, OpenAI


@app.cell
def _(OPENAI_API_KEY, OpenAI):
    client = OpenAI(api_key=OPENAI_API_KEY)
    return (client,)


@app.cell
def _(client, mo):
    def _():
        prompt_input = """Write a coincise message to remind users 
        to be vigilant about phishing attacks."""
        response = client.chat.completions.create(
          model="gpt-5-nano",
          messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_input}
          ] 
        )
    
        return mo.Html(f"{response.choices[0].message.content}")
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Running prompts with LangChain
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Basic prompt with LangChain
    """)
    return


@app.cell
def _(OPENAI_API_KEY):
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                     model_name="gpt-5-nano")
    return (llm,)


@app.cell
def _(llm, mo):
    def _():
        prompt_input = """Write a coincise message to remind users 
                        to be vigilant about phishing attacks."""

        response = llm.invoke(prompt_input)
        return mo.Html(f"{response.content}")
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prompt templates
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Prompt template - implementing it with a Python function
    """)
    return


@app.function
def generate_text_summary_prompt(text, num_words, tone):
    return f"You are an experienced copywriter. Write a {num_words} words summary the the following text, using a {tone} tone: {text}"


@app.cell
def _():
    segovia_aqueduct_text = """The Aqueduct of Segovia (Spanish: 
    Acueducto de Segovia) is a Roman aqueduct in Segovia, Spain. 
    It was built around the first century AD to channel water from 
    springs in the mountains 17 kilometres (11 mi) away to the 
    city's fountains, public baths and private houses, and was in 
    use until 1973. 
    Its elevated section, with its complete arcade of 167 arches, 
    is one of the best-preserved Roman aqueduct bridges and the 
    foremost symbol of Segovia, as evidenced by its presence on the 
    city's coat of arms. 
    The Old Town of Segovia and the aqueduct, were declared a UNESCO 
    World Heritage Site in 1985. As the aqueduct lacks a legible 
    inscription (one was apparently located in the structure's attic, 
    or top portion[citation needed]), the date of construction cannot be 
    definitively determined. The general date of the Aqueduct's 
    construction was long a mystery, although it was thought to have 
    been during the 1st century AD, during the reigns of the Emperors 
    Domitian, Nerva, and Trajan. At the end of the 20th century, 
    Géza Alföldy deciphered the text on the dedication plaque by 
    studying the anchors that held the now missing bronze letters 
    in place. He determined that Emperor Domitian (AD 81–96) ordered 
    its construction[1] and the year 98 AD was proposed as the most 
    likely date of completion.[2] However, in 2016 archeological 
    evidence was published which points to a slightly later date, 
    after 112 AD, during the government of Trajan or in the 
    beginning of the government of emperor Hadrian, 
    from 117 AD."""
    return (segovia_aqueduct_text,)


@app.cell
def _(llm, mo, segovia_aqueduct_text):
    def _():
        input_prompt = generate_text_summary_prompt(
            text=segovia_aqueduct_text, 
            num_words=20,
            tone="knowledgeable and engaging")

        response = llm.invoke(input_prompt)
        return mo.Html(f"{response.content}")
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prompt template - using LangChain's ChatPromptTemplate
    """)
    return


@app.cell
def _():
    from langchain_core.prompts import PromptTemplate

    prompt_template = PromptTemplate.from_template(
    """You are an experienced copywriter. 
    Write a {num_words} words summary the the following text, 
    using a {tone} tone: {text}""")
    return (prompt_template,)


@app.cell
def _(llm, mo, prompt_template, segovia_aqueduct_text):
    def _():
        prompt = prompt_template.format(
            text=segovia_aqueduct_text,
            num_words=20, 
            tone="knowledgeable and engaging")

        response = llm.invoke(prompt)
        return mo.Html(f"{response.content}")

    _()
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    mo.Html(f"{A more reliable way to enforce prompt structure—recommended in both OpenAI’s
    and Anthropic’s prompt engineering guidelines (see the resource list at the end of
    this section)—is to mark different sections of the prompt with XML-style tags. For
    example, the previous prompt could be written as
    <Persona>
    You’re an experienced Large Language Model (LLM) developer and renowned
    speaker.
    </Persona>
    <Context>
    You’ve been invited to give a keynote speech for a LLM event.
    </Context>50 CHAPTER 2 Executing prompts programmatically<Instruction>
    Write the punch lines for the speech.
    </Instruction>
    <Input>
    Include the following facts: LLMs have become mainstream with the launch of ChatGPT in November 2022 many popular LLMs and LLM based chatbots have been launched since then, such
    as LLAMA-2, Falcon180B, Bard. LLMs becoming as popular as search engines many companies want to integrate LLMs in their applications
    </Input>
    <Tone>
    Use a witty but entertaining tone.
    </Tone>
    <OutputFormat>
    Present the text in two paragraphs of 5 lines each.
    </OutputFormat>}")
    """)
    return


if __name__ == "__main__":
    app.run()
