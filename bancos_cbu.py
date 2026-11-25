"""Banco a partir del código de entidad BCRA (3 primeros dígitos de la CBU)."""

CODIGOS_ENTIDAD = {
    "007": "Galicia",
    "011": "Nación",
    "014": "Provincia",
    "015": "ICBC",
    "016": "Citibank",
    "017": "BBVA",
    "020": "Córdoba",
    "027": "Supervielle",
    "029": "Ciudad",
    "034": "Patagonia",
    "044": "Hipotecario",
    "045": "San Juan",
    "072": "Santander",
    "083": "Chubut",
    "086": "Santa Cruz",
    "093": "La Pampa",
    "094": "Corrientes",
    "097": "Provincia del Neuquén",
    "143": "Brubank",
    "150": "HSBC",
    "191": "Credicoop",
    "259": "Itaú",
    "268": "Tierra del Fuego",
    "285": "Macro",
    "299": "Comafi",
    "301": "Piano",
    "309": "Rioja",
    "311": "Nuevo Banco del Chaco",
    "315": "Formosa",
    "321": "Santiago del Estero",
    "322": "Industrial",
    "330": "Nuevo Banco de Santa Fe",
    "384": "Wilobank",
    "386": "Nuevo Banco de Entre Ríos",
    "415": "Reba",
    "426": "Bica",
    "453": "NaranjaX",
}


# CVU: "000" + código de PSP (4 dígitos). Solo prefijos verificados contra imágenes reales.
CODIGOS_PSP = {
    "0000006": "Administradora San Juan SA",
    "0000013": "Prex",
    "0000076": "Personal Pay",
    "0000085": "Claro Pay",
    "0000139": "Belo",
    "0000168": "Lemon",
}


def billetera_por_cvu(cvu: str) -> str:
    if len(cvu) != 22 or not es_cvu(cvu):
        return ""
    return CODIGOS_PSP.get(cvu[:7], "")


def es_cvu(numero: str) -> bool:
    """Las CVU de billeteras virtuales empiezan con 000."""
    return len(numero) == 22 and numero.startswith("000")


def banco_por_cbu(cbu: str) -> str:
    if len(cbu) != 22 or not cbu.isdigit():
        return ""
    return CODIGOS_ENTIDAD.get(cbu[:3], "")


if __name__ == "__main__":
    assert banco_por_cbu("0110031030003121658665") == "Nación"
    assert banco_por_cbu("0970003155003065660019") == "Provincia del Neuquén"
    assert banco_por_cbu("9990000000000000000000") == ""
    assert banco_por_cbu("N/A") == ""
    assert es_cvu("0000076500000044198183") and not es_cvu("0110031030003121658665")
    assert billetera_por_cvu("0000013000032187926736") == "Prex"
    assert billetera_por_cvu("0110031030003121658665") == ""
    print("ok")
