"""Repair documented archive renaming without modifying EEG or marker payloads."""
from pathlib import Path
import re
from .io import sha256_file


def repair_links(header):
    header=Path(header)
    raw=header.read_bytes()
    encoding="utf-8-sig"
    try: text=raw.decode(encoding)
    except UnicodeDecodeError:
        encoding="cp1252"; text=raw.decode(encoding)
    changes=[]
    for key,suffix in (("DataFile",".eeg"),("MarkerFile",".vmrk")):
        match=re.search(r"(?m)^"+key+r"=(.*)\r?$",text)
        if match is None: raise ValueError(f"BrainVision {key} declaration missing")
        declared=match.group(1).strip().replace("\\","/")
        original=(header.parent/declared).resolve()
        if original.is_file(): continue
        candidates=sorted(p for p in header.parent.iterdir() if p.suffix.lower()==suffix and p.is_file())
        exact=[p for p in candidates if p.stem.lower()==header.stem.lower()]
        if len(exact)==1: chosen=exact[0]
        elif len(candidates)==1: chosen=candidates[0]
        else:
            raise FileNotFoundError(f"Cannot uniquely resolve {key}: {declared}; same-directory candidates={[p.name for p in candidates]}")
        text=text[:match.start()]+key+"="+chosen.name+text[match.end():]
        changes.append({"key":key,"declared":declared,"resolved":chosen.name,"payload_sha256":sha256_file(chosen)})
    result=header.with_name(header.stem+"_resolved.vhdr")
    result.write_bytes(text.encode(encoding))
    return result,{"original_header":header.name,"original_sha256":sha256_file(header),
        "temporary_header":result.name,"temporary_sha256":sha256_file(result),"changes":changes}
