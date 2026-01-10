from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from src.agents.agent_state import AgentState
from src.tools.calculator import calculate_clinical_score # A tool refatorada
import json

# Carrega variáveis
from dotenv import load_dotenv
load_dotenv()

# 1. Configuração Correta do LLM com Tools
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# Tech Lead Tip: Salve o objeto com bind em uma variável nova!
llm_with_tools = llm.bind_tools([calculate_clinical_score])

def Mathematician_node(state: AgentState) -> AgentState:
    """
    O Nó Matemático:
    1. Analisa os vitais extraídos.
    2. Decide qual score é pertinente (ou ambos).
    3. Invoca a tool determinística (Python).
    4. Atualiza o estado com o resultado auditável.
    """
    
    # Recupera dados do Estado (Output do Scribe)
    data = state.get("extracted_data")
    if not data or not data.clinical or not data.clinical.vitals:
        return AgentState(
            input=state["input"],
            extracted_data=state.get("extracted_data"),
            validation_error=state.get("validation_error"),
            attempts=state.get("attempts", 0),
            risk_score_report="Sem dados clínicos válidos para calcular scores."
        )
    
    vitals = data.clinical.vitals
    
    # 2. Prompt Focado em AÇÃO (Tool Call), não em Cálculo Mental
    system_msg = SystemMessage(content="""
    Você é um assistente clínico rigoroso. Sua única função é calcular scores de risco usando a ferramenta disponível.
    NÃO calcule nada de cabeça.
    1. Analise os sinais vitais fornecidos.
    2. Chame a ferramenta 'calculate_clinical_score' para 'MEWS' e/ou 'NEWS'.
    3. Se houver dados suficientes, calcule ambos.
    """)
    
    # Serializamos o objeto Pydantic para JSON para o LLM entender a estrutura
    # Isso é mais limpo que fazer f-strings manuais
    vitals_json = vitals.model_dump_json()
    
    user_msg = HumanMessage(content=f"Sinais Vitais do Paciente: {vitals_json}")
    
    # 3. Invocação do Modelo
    response = llm_with_tools.invoke([system_msg, user_msg])
    
    # 4. Execução da Tool (Manual Execution para controle total)
    # No LangGraph, poderíamos ter um nó separado 'ToolNode', mas para TCC, 
    # fazer in-line aqui mostra controle sobre o fluxo.
    
    tool_outputs = []
    
    if response.tool_calls:
        print(f"DEBUG: O Agente decidiu chamar {len(response.tool_calls)} ferramentas.")
        
        for tool_call in response.tool_calls:
            # O LangChain já parseou os argumentos para nós
            tool_args = tool_call["args"]
            
            # Precisamos reconstruir o objeto VitalsSchema para passar para a função Python
            # pois o JSON veio como dict
            try:
                # Importe sua classe VitalsSchema aqui
                from src.schemas.scribe_output_schema import VitalsSchema 
                vitals_obj = VitalsSchema(**tool_args['vitals'])
                score_name = tool_args['score_name']
                
                # Executa a função Python determinística
                result_str = calculate_clinical_score.invoke({"vitals": vitals_obj, "score_name": score_name})
                
                tool_outputs.append(result_str)
                
            except Exception as e:
                tool_outputs.append(f"Erro ao executar cálculo: {str(e)}")
    else:
        print("DEBUG: O Agente não quis chamar nenhuma ferramenta.")
        tool_outputs.append("Nenhum score calculado (dados insuficientes ou decisão do agente).")

    # 5. Atualização do Estado
    # Unificamos os resultados em uma string consolidada para o próximo agente (Auditor)
    final_score_report = "\n".join(tool_outputs)
    
    # Retornamos apenas o delta do estado que queremos atualizar
    return AgentState(
        input=state["input"],
        extracted_data=state.get("extracted_data"),
        validation_error=state.get("validation_error"),
        attempts=state.get("attempts", 0),
        risk_score_report=final_score_report
    )