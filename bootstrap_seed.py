# bootstrap_seed.py
import json
from src.config import path

QUESTIONS = [
    # Reformation
    {"qid":"q001","question":"Who wrote the Ninety-five Theses?","answer":"Martin Luther","supporting_chunks":[],"hop_type":"single","question_type":"short"},
    {"qid":"q002","question":"Did Martin Luther write the Ninety-five Theses?","answer":"yes","supporting_chunks":[],"hop_type":"single","question_type":"yesno"},
    {"qid":"q003","question":"Which reformer was based in Geneva?","answer":"John Calvin","supporting_chunks":[],"hop_type":"single","question_type":"short"},
    {"qid":"q004","question":"Was the Council of Trent part of the Counter-Reformation?","answer":"yes","supporting_chunks":[],"hop_type":"single","question_type":"yesno"},
    {"qid":"q005","question":"Who was the monk whose challenge to indulgences triggered the Protestant Reformation?","answer":"Martin Luther","supporting_chunks":[],"hop_type":"multi","question_type":"short"},

    # Islamic Golden Age
    {"qid":"q006","question":"Which caliphate sponsored the House of Wisdom in Baghdad?","answer":"Abbasid","supporting_chunks":[],"hop_type":"multi","question_type":"short"},
    {"qid":"q007","question":"Who was the mathematician at the House of Wisdom known for founding algebra?","answer":"Al-Khwarizmi","supporting_chunks":[],"hop_type":"multi","question_type":"short"},
    {"qid":"q008","question":"Was Baghdad the capital of the Abbasid Caliphate?","answer":"yes","supporting_chunks":[],"hop_type":"single","question_type":"yesno"},
    {"qid":"q009","question":"What mathematical field is Al-Khwarizmi known for founding?","answer":"algebra","supporting_chunks":[],"hop_type":"single","question_type":"short"},
    {"qid":"q010","question":"Which Abbasid caliph expanded the House of Wisdom and supported Al-Khwarizmi?","answer":"al-Ma'mun","supporting_chunks":[],"hop_type":"multi","question_type":"short"},

    # Crusades
    {"qid":"q011","question":"Which council called for the First Crusade?","answer":"Council of Clermont","supporting_chunks":[],"hop_type":"single","question_type":"short"},
    {"qid":"q012","question":"Did the First Crusade capture Jerusalem?","answer":"yes","supporting_chunks":[],"hop_type":"single","question_type":"yesno"},
    {"qid":"q013","question":"What was the Iberian campaign that paralleled the Crusades called?","answer":"Reconquista","supporting_chunks":[],"hop_type":"multi","question_type":"short"},
    {"qid":"q014","question":"Was the First Crusade launched before or after the Council of Clermont?","answer":"after","supporting_chunks":[],"hop_type":"multi","question_type":"short"},

    # Buddhism / Silk Road
    {"qid":"q015","question":"Which Maurya emperor promoted the spread of Buddhism?","answer":"Ashoka","supporting_chunks":[],"hop_type":"single","question_type":"short"},
    {"qid":"q016","question":"Was Ashoka a Maurya emperor?","answer":"yes","supporting_chunks":[],"hop_type":"single","question_type":"yesno"},
    {"qid":"q017","question":"What trade network helped spread Buddhism from India to China?","answer":"Silk Road","supporting_chunks":[],"hop_type":"single","question_type":"short"},
    {"qid":"q018","question":"Which emperor's missionaries used the Silk Road to spread Buddhism?","answer":"Ashoka","supporting_chunks":[],"hop_type":"multi","question_type":"short"},
    {"qid":"q019","question":"Did Buddhism spread from India to China along the Silk Road?","answer":"yes","supporting_chunks":[],"hop_type":"multi","question_type":"yesno"},
    {"qid":"q020","question":"Which emperor convened the Third Buddhist Council?","answer":"Ashoka","supporting_chunks":[],"hop_type":"multi","question_type":"short"},
]

def main():
    qpath = path("questions")
    qpath.parent.mkdir(parents=True, exist_ok=True)
    with open(qpath, "w", encoding="utf-8") as f:
        for q in QUESTIONS:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Wrote {len(QUESTIONS)} seed questions -> {qpath}")

    sdir = path("splits")
    sdir.mkdir(parents=True, exist_ok=True)
    ids = [q["qid"] for q in QUESTIONS]
    dev, test = ids[:12], ids[12:]
    json.dump(dev,  open(sdir / "dev.json",  "w"))
    json.dump(test, open(sdir / "test.json", "w"))
    print(f"dev={len(dev)} test={len(test)} -> {sdir}")

if __name__ == "__main__":
    main()