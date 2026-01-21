def normalize_vitals_for_calculator(extraction_json: dict) -> dict:
    """
    Adapter Robusto (Polimórfico).
    Aceita tanto o formato 'Rich JSON' (aninhado) quanto o 'Flat JSON' (simples).
    Retorna um dicionário plano pronto para a ferramenta de cálculo (Calculator Tool).
    """

    # 1. Try to locate where the vitals are
    source = extraction_json.get("vital_signs") or extraction_json.get("extracted_vitals") or extraction_json

    def _get_val(keys_list, target_type=float):
        """Helper para buscar valor em múltiplas chaves possíveis e converter tipo."""
        for key in keys_list:
            val = None
            # Se for dicionário aninhado (Rich), tenta pegar .value ou .normalized_value_celsius
            if isinstance(source.get(key), dict):
                val = source[key].get("normalized_value_celsius") or source[key].get("value")
            # Se for valor direto (Flat)
            else:
                val = source.get(key)
            
            if val is not None:
                try:
                    return target_type(val)
                except (ValueError, TypeError):
                    continue # Tenta a próxima chave se a conversão falhar
        return None

    def _get_bp(component):
        """Helper específico para pressão arterial."""
        # Tenta estrutura aninhada: blood_pressure -> systolic
        bp_obj = source.get("blood_pressure")
        if isinstance(bp_obj, dict):
            return bp_obj.get(component)
        # Tenta chaves planas: sbp, systolic, etc.
        return _get_val([component, "sbp" if component == "systolic" else "dbp"])

    # 2. Mapeamento e Normalização
    normalized = {
        "heartrate": _get_val(["heartrate", "heart_rate", "hr"]),
        "resprate": _get_val(["resprate", "respiratory_rate", "rr"]),
        "temperature": _get_val(["temperature", "temp", "t"]),
        "o2sat": _get_val(["o2sat", "oxygen_saturation", "spo2", "sat"]),
        "sbp": _get_bp("systolic"),
        "dbp": _get_bp("diastolic"),
        "gcs": _get_val(["gcs", "glasgow"], int),
        "avpu": source.get("avpu") or source.get("acvpu") # AVPU geralmente é string
    }

    # 3. Lógica de Oxigênio Suplementar (Crucial para NEWS2)
    # Se não detectado explicitamente, assume False (segurança)
    supp_o2 = source.get("supplemental_oxygen")
    if isinstance(supp_o2, dict):
        # Lógica para extração complexa: "Room Air" -> False
        delivery = str(supp_o2.get("delivery_method", "")).lower()
        normalized["supplemental_oxygen"] = False if "room" in delivery or "ambient" in delivery else True
    else:
        normalized["supplemental_oxygen"] = bool(supp_o2) if supp_o2 is not None else False

    return normalized

def check_data_sufficiency(normalized_data: dict, score_type: str = "NEWS") -> list:
    """
    Função Auxiliar para o 'Auditor Agent'.
    Retorna lista de campos faltantes antes mesmo de tentar calcular.
    """
    required = {
        "NEWS": ["resprate", "o2sat", "sbp", "heartrate", "temperature", "avpu"], # AVPU pode ser inferido de GCS
        "MEWS": ["resprate", "heartrate", "sbp", "temperature", "avpu"]
    }
    
    missing = []
    req_fields = required.get(score_type, [])
    
    for field in req_fields:
        if normalized_data.get(field) is None:
            # Fallback lógico: Se falta AVPU mas tem GCS, aceita.
            if field == "avpu" and normalized_data.get("gcs") is not None:
                continue
            missing.append(field)
            
    return missing