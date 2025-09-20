results=[]
for i in range(12):
    x=int(input(f"enter la valeur de MRE du mois {i+1}/2010: "))
    y=int(input(f"enter la valeur de TES du mois {i+1}/2010: "))
    results.append({
        "date":f"{i+1}/2010",
        "MRE":x,
        "TRS":y
    })

