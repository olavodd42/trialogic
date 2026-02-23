import argparse
from src.main import create_workflow
from src.schemas.input_schema import InputSchema
import logging

# Configura o log para aparecer no terminal
logging.basicConfig(level=logging.INFO)

# Um caso que sabemos que é Sepsis (deve ativar o RAG se tiver docs de Sepsis)
# Ou Trauma, se você tiver docs de Trauma.
MOCK_TEXT = """
Triage Note: 55y male presents with fever and hypotension.
History of UTI. 
Vitals: T 38.9, HR 115, BP 85/50, RR 22, SpO2 94% on RA.
Alert but lethargic.
"""

def test_rag_integration():
    parser = argparse.ArgumentParser(description="TriaLogic Single Case Runner")
    parser.add_argument("--no-validator", action="store_true", help="Disable Validator node.")
    parser.add_argument("--no-rag", action="store_true", help="Disable Clinical RAG and Synthesizer.")
    parser.add_argument("--probabilistic", action="store_true", help="Use LLM-based probabilistic Mathematician.")
    args = parser.parse_args()

    print("🚀 INICIANDO TESTE UNITÁRIO DO PIPELINE...")
    app = create_workflow(
        use_validator=not args.no_validator,
        use_rag=not args.no_rag,
        use_probabilistic=args.probabilistic,
    )
    
    input_data = InputSchema(
        subject_id=1,
        hadm_id=1,
        raw_text=MOCK_TEXT
    )
    
    result = app.invoke({"input": input_data})
    
    print("\n" + "="*40)
    print("📊 RESULTADO DO TESTE")
    print("="*40)
    
    # 1. Verifica RAG
    rag_used = result.get("rag_context_used", False)
    auditor = result.get("auditor_report", {})
    quote = auditor.get("evidence_quote", "")
    
    print(f"1. RAG Ativado? {'✅ Sim' if rag_used else '❌ Não'}")
    
    if len(str(quote)) > 10:
        print(f"2. Citação Encontrada: ✅ \"{quote[:100]}...\"")
    else:
        print(f"2. Citação: ⚠️ Vazia (O RAG funcionou mas não achou docs, ou falhou silenciosamente).")
        
    # 3. Verifica Diagnóstico
    print(f"3. Veredito: {auditor.get('clinical_risk_category')}")
    print(f"4. Sugestão: {auditor.get('clinical_suggestion')}")

if __name__ == "__main__":
    test_rag_integration()