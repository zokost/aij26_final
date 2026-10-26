import {serializeProject,deserializeProject,projectBinarySize,archiveDatabase} from './review-store.mjs?v=features-20261019c';

export const CASE_FORMAT_VERSION=1;
const FORMAT='detective-case',DB='detective-cases',STORE='cases',MAX_BINARY=250*1024*1024,MAX_JSON=300*1024*1024;
const fail=()=>{throw new Error('Обращение повреждено или превышает ограничения формата.');};
const text=(value,max)=>typeof value==='string'&&value.length<=max?value:fail();
const decisions=['pending','review','accepted'];

function metadata(value){
  if(!value||typeof value!=='object'||Array.isArray(value)||!decisions.includes(value.decision))fail();
  const result={id:text(value.id,128),title:text(value.title,200),note:text(value.note,10000),decision:value.decision};
  if(!result.id)fail();
  return result;
}

function jsonSize(value){
  let binaryText=0;
  const metadata=JSON.stringify(value,(key,entry)=>{
    if((key==='base64'||key==='mask_png_base64')&&typeof entry==='string'){binaryText+=entry.length;return '';}
    return entry;
  });
  return new Blob([metadata]).size+binaryText;
}

function allItems(items){
  return items.flatMap(item=>[item,...(Array.isArray(item.videoFrames)?item.videoFrames:[])]);
}

function uniqueItems(items){
  if(!Array.isArray(items)||items.length>20)fail();
  const ids=new Set();
  for(const item of allItems(items)){
    if(!item||typeof item!=='object')fail();
    const id=text(item.id,128);if(!id||ids.has(id))fail();ids.add(id);
  }
}

export function caseSummary({items=[]}={}){
  const checks=allItems(items),videos=items.filter(item=>item.kind==='video');
  const decision=item=>item.reviewDraft?.decision||item.review?.decision||'pending';
  const count=predicate=>checks.filter(predicate).length;
  const summary={materials:items.length,checks:checks.length,analysed:count(item=>!!item.analysis),
    unanalysed:count(item=>item.kind==='image'&&!item.analysis),unresolved:count(item=>decision(item)==='pending'),
    findings:count(item=>item.analysis?.area_frac>0||item.analysis?.n_regions>0||item.regions?.length>0),
    failed:count(item=>!!item.error||item.videoScan?.status==='error'),
    cancelled:count(item=>!!item.cancelled||item.videoScan?.status==='cancelled'),
    partial_videos:videos.filter(item=>item.videoScan?.status!=='complete').length,
    review_requested:count(item=>decision(item)==='review')};
  summary.status=!items.length?'empty':checks.some(item=>item.pending||item.queued||item.videoScan?.status==='running')?'in_progress':
    summary.unresolved||summary.review_requested||summary.failed||summary.cancelled||summary.partial_videos?'needs_review':'reviewed';
  return summary;
}

function runtimeBytes(item){
  let size=(item.file||item.originalFile)?.size||0;
  size+=item.reference?.file?.size||0;
  if(item.analysis?.mask_png_base64){const encoded=item.analysis.mask_png_base64;size+=encoded.length/4*3-(encoded.endsWith('==')?2:encoded.endsWith('=')?1:0);}
  if(item.videoFrames!==undefined){
    if(!Array.isArray(item.videoFrames)||item.videoFrames.length>12)fail();
    for(const frame of item.videoFrames){if(frame.videoFrames!==undefined)fail();size+=runtimeBytes(frame);}
  }
  return size;
}

export async function serializeCase(value,getReview){
  const info=metadata(value),items=value.items;
  uniqueItems(items);
  if(items.reduce((sum,item)=>sum+runtimeBytes(item),0)>MAX_BINARY)fail();
  // Every serializeProject snapshots its metadata before its first await.
  const projects=await Promise.all(items.map(item=>serializeProject(item,getReview?getReview(item):item.reviewDraft)));
  if(projects.reduce((sum,project)=>sum+projectBinarySize(project),0)>MAX_BINARY)fail();
  const archive={format:FORMAT,format_version:CASE_FORMAT_VERSION,saved_at:new Date().toISOString(),case:info,projects};
  if(jsonSize(archive)>MAX_JSON)fail();
  return archive;
}

export async function deserializeCase(value){
  if(jsonSize(value)>MAX_JSON)fail();
  if(!value||typeof value!=='object'||value.format!==FORMAT||value.format_version!==CASE_FORMAT_VERSION||
    typeof value.saved_at!=='string'||value.saved_at.length>64||!Number.isFinite(Date.parse(value.saved_at))||
    !Array.isArray(value.projects)||value.projects.length>20)fail();
  const info=metadata(value.case),saved_at=value.saved_at;
  if(value.projects.reduce((sum,project)=>sum+projectBinarySize(project),0)>MAX_BINARY)fail();
  const items=[];
  for(const project of value.projects)items.push(await deserializeProject(project));
  uniqueItems(items);
  return {...info,items,saved_at,restored:true};
}

const database=(action,mode)=>archiveDatabase(DB,STORE,action,mode);

export async function saveCase(project){
  if(jsonSize(project)>MAX_JSON)fail();
  const archive=structuredClone(project),value=await deserializeCase(archive);
  const summary={...metadata(value),saved_at:value.saved_at,...caseSummary(value)};
  await database(store=>store.put({id:value.id,summary,archive}),'readwrite');
  return summary;
}

export async function listCases(){
  const summaries=await database(store=>{
    const rows=[],request=store.openCursor();
    request.onsuccess=()=>{const cursor=request.result;if(cursor){rows.push(cursor.value.summary);cursor.continue();}};
    return {result:rows};
  });
  return summaries.sort((a,b)=>b.saved_at.localeCompare(a.saved_at));
}

export async function loadCase(id){
  const record=await database(store=>store.get(id));
  return record?deserializeCase(record.archive):null;
}

export const deleteCase=id=>database(store=>store.delete(id),'readwrite');
export const clearCases=()=>database(store=>store.clear(),'readwrite');
