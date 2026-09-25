"""Recorded tool-disabled LLM calls; typed JSON parsing never substitutes for a live call."""
from pathlib import Path
import json, re, subprocess, tempfile, time, uuid, socket

MODEL='claude-sonnet-5'
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def parse(text):
    def pairs(items):
        out={}
        for k,v in items:
            if k in out:raise ValueError('duplicate key')
            out[k]=v
        return out
    text=text.strip();m=re.fullmatch(r'```(?:json)?\s*\n(.*)\n```',text,re.S)
    return json.loads(m[1] if m else text,object_pairs_hook=pairs,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite')))
def call(folder,prompt,system='Use only the supplied evidence. Return the requested JSON. Do not use tools.',timeout=300):
    folder.mkdir(parents=True,exist_ok=False)
    record=dict(host=socket.gethostname(),model_requested=MODEL,started=time.time(),prompt=prompt,system=system,status='error')
    save(folder/'request.json',record)
    with tempfile.TemporaryDirectory(prefix='regional-llm-') as cwd:
        cmd=['claude','-p',prompt,'--model',MODEL,'--output-format','json','--tools','',
             '--strict-mcp-config','--mcp-config','{"mcpServers":{}}','--disable-slash-commands',
             '--system-prompt',system,'--session-id',str(uuid.uuid4())]
        try:
            p=subprocess.run(cmd,cwd=cwd,text=True,capture_output=True,timeout=timeout)
            record.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
            outer=json.loads(p.stdout)
            if p.returncode or outer.get('is_error') or outer.get('subtype')!='success':raise ValueError('CLI did not complete successfully')
            record.update(status='ok',text=outer['result'],models_reported=list(outer.get('modelUsage',{})),cli_result=outer)
        except Exception as exc:record.update(error=repr(exc))
    record['finished']=time.time();save(folder/'response.json',record)
    return record
