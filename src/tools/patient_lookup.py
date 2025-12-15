# src/tools/patient_lookup.py
from langchain_core.tools import tool
from src.utils.database import get_db_connection

@tool
def fetch_patient_triage_data(stay_id: int) -> str:
    """
    Consulta a base de dados clínica para obter os dados vitais e queixas
    iniciais de um paciente baseado no seu ID de estadia (stay_id).
    
    Args:
        stay_id: O identificador único da admissão na emergência.
        
    Returns:
        String contendo temperatura, ritmo cardíaco, PA e queixa principal.
    """
    db = get_db_connection()
   
    query = f"""
    SELECT triage_temperature, triage_heartrate, triage_resprate, triage_o2sat, triage_sbp, triage_dbp, chiefcomplaint
    FROM ed_triage
    WHERE stay_id = {stay_id}
    LIMIT 1;
    """
    
    try:
        result = db.run(query)
        if not result:
            return "Nenhum registo de triagem encontrado para este ID."
        return f"Dados de Triagem: {result}"
    except Exception as e:
        return f"Erro ao consultar base de dados: {str(e)}"