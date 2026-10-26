import {modelResult,operationLabels} from './data.mjs?v=features-20261019c';

export const PROJECT_FORMAT_VERSION=1;
const FORMAT='detective-review',DB='detective-reviews',STORE='reviews',MB=1024*1024;
const IMAGE_TYPES=['image/jpeg','image/png','image/webp'],VIDEO_TYPES=['video/mp4','video/webm'];
const fail=()=>{throw new Error('Файл проекта повреждён или имеет неподдерживаемый формат.');};
const object=value=>value&&typeof value==='object'&&!Array.isArray(value)?value:fail();
const text=(value,max=1000)=>typeof value==='string'&&value.length<=max?value:fail();
const number=(value,min,max)=>typeof value==='number'&&Number.isFinite(value)&&value>=min&&value<=max?value:fail();
const integer=(value,min,max)=>Number.isInteger(value)?number(value,min,max):fail();
const boolean=value=>typeof value==='boolean'?value:fail();
const sha=value=>/^[a-f0-9]{64}$/.test(value)?value:fail();
const date=value=>Number.isFinite(Date.parse(text(value,64)))?value:fail();
const optionalText=value=>value==null?null:text(value);
const hash=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');

function bytesFromBase64(value,maxBytes){
  text(value,Math.ceil(maxBytes/3)*4);
  if(!value.length||value.length%4||!/^[A-Za-z0-9+/]*={0,2}$/.test(value))fail();
  const raw=atob(value);
  if(raw.length>maxBytes||btoa(raw)!==value)fail();
  return Uint8Array.from(raw,c=>c.charCodeAt(0));
}

async function base64(blob){
  const bytes=new Uint8Array(await blob.arrayBuffer());
  let raw='';
  for(let i=0;i<bytes.length;i+=32768)raw+=String.fromCharCode(...bytes.subarray(i,i+32768));
  return btoa(raw);
}

function dimensions(value){
  const width=integer(value.width,1,15_000_000),height=integer(value.height,1,15_000_000);
  if(width*height>15_000_000)fail();
  return {width,height};
}

async function media(original,kind,filename,width,height){
  object(original);
  if(!(kind==='image'?IMAGE_TYPES:VIDEO_TYPES).includes(original.type))fail();
  const bytes=bytesFromBase64(original.base64,(kind==='video'?100:20)*MB),inputHash=await hash(bytes);
  if(sha(original.sha256)!==inputHash)fail();
  const ascii=(start,end)=>String.fromCharCode(...bytes.slice(start,end));
  if(original.type==='image/png'&&(bytes.length<24||[137,80,78,71,13,10,26,10].some((b,i)=>bytes[i]!==b)||
      new DataView(bytes.buffer).getUint32(16)!==width||new DataView(bytes.buffer).getUint32(20)!==height)||
    original.type==='image/jpeg'&&!(bytes[0]===255&&bytes[1]===216&&bytes[2]===255)||
    original.type==='image/webp'&&!(ascii(0,4)==='RIFF'&&ascii(8,12)==='WEBP')||
    original.type==='video/mp4'&&ascii(4,8)!=='ftyp'||
    original.type==='video/webm'&&![26,69,223,163].every((b,i)=>bytes[i]===b))fail();
  const file=new File([bytes],filename,{type:original.type,lastModified:integer(original.last_modified,0,1e16)});
  if(typeof createImageBitmap==='function'&&kind==='image'){
    const bitmap=await createImageBitmap(file);
    try{if(bitmap.width!==width||bitmap.height!==height)fail();}finally{bitmap.close();}
  }
  return {file,inputHash};
}

async function encodeFile(file,kind){
  if(!file||file.size>(kind==='video'?100:20)*MB||!(kind==='video'?VIDEO_TYPES:IMAGE_TYPES).includes(file.type))fail();
  return {type:file.type,base64:await base64(file),sha256:await hash(await file.arrayBuffer()),last_modified:file.lastModified??0};
}

function snapshot(item,operatorReview,child=false){
  if(!item||item.pending||item.error||item.explanationPending||item.videoScan?.status==='running')throw new Error('Дождитесь завершения проверки перед сохранением.');
  if(child&&(item.kind!=='image'||item.videoFrames!==undefined||item.videoScan!==undefined))fail();
  const result=structuredClone({id:item.id,filename:item.filename,kind:item.kind,source:item.source,width:item.width,height:item.height,
    analysis:item.analysis??null,regions:item.regions||[],explanations:item.explanations||{},review:operatorReview,
    duration:item.duration,video_time:item.video_time,captured_at:item.captured_at,parentVideoId:item.parentVideoId,videoScan:item.videoScan});
  result.file=item.file||item.originalFile;
  if(item.reference)result.reference={file:item.reference.file,...dimensions(item.reference)};
  if(item.videoFrames!==undefined){
    if(!Array.isArray(item.videoFrames)||item.videoFrames.length>12)fail();
    result.videoFrames=item.videoFrames.map(frame=>snapshot(frame,frame.reviewDraft||{decision:'pending',note:''},true));
    if(result.videoFrames.reduce((sum,frame)=>sum+(frame.file?.size||0),0)>50*MB)fail();
  }
  return result;
}

function scan(value,duration,frames){
  object(value);
  if(!['complete','cancelled','error'].includes(value.status)||!Array.isArray(value.planned_times)||value.planned_times.length>12)fail();
  const times=value.planned_times.map(t=>number(t,0,duration));
  if(times.some(t=>t>=duration)||new Set(times).size!==times.length||frames.length>times.length)fail();
  if(value.status==='complete'&&(!times.length||frames.length!==times.length||frames.some(frame=>!frame.analysis)))fail();
  return {status:value.status,planned_times:times,...(value.error!==undefined?{error:text(value.error,5000)}:{})};
}

function model(value){
  object(value);
  const result={checkpoint:text(value.checkpoint),sha256:sha(value.sha256),threshold:number(value.threshold,0,1),
    precision:text(value.precision,32),device:text(value.device,64)};
  for(const key of ['architecture','encoder','experiment','score_kind'])if(value[key]!==undefined)result[key]=text(value[key]);
  for(const key of ['refine','square','score_calibrated','matmul_tf32','cudnn_tf32'])if(value[key]!==undefined)result[key]=boolean(value[key]);
  for(const key of ['decoder_channels','img_size'])if(value[key]!==undefined)result[key]=value[key]===null?null:integer(value[key],1,16384);
  for(const key of ['canvases','active_canvases'])if(value[key]!==undefined){
    if(!Array.isArray(value[key])||!value[key].length||value[key].length>16)fail();
    result[key]=value[key].map(c=>{
      if(!Array.isArray(c)||c.length!==2)fail();
      const {width,height}=dimensions({height:c[0],width:c[1]});return [height,width];
    });
  }
  if(result.canvases&&result.active_canvases?.some(c=>!result.canvases.some(a=>a[0]===c[0]&&a[1]===c[1])))fail();
  return result;
}

function review(value){
  object(value);
  if(!['pending','review','accepted'].includes(value.decision))fail();
  return {decision:value.decision,note:text(value.note,20000)};
}

function rectangle(value,width,height){
  const x=number(value.x,0,width),y=number(value.y,0,height),w=number(value.width,1,width),h=number(value.height,1,height);
  if(x+w>width||y+h>height)fail();
  return {x,y,width:w,height:h};
}

function manualRegions(values,width,height,modelIds){
  if(!Array.isArray(values)||values.length>1000)fail();
  const ids=new Set(modelIds);
  return values.map(value=>{
    object(value);const id=integer(value.id,1,1_000_000);
    if(ids.has(id)||value.manual!==true||!Array.isArray(value.polygon)||value.polygon.length<3||value.polygon.length>1000)fail();
    ids.add(id);
    return {id,manual:true,annotation_source:'manual',title:text(value.title),hint:text(value.hint,5000),
      polygon:value.polygon.map(p=>Array.isArray(p)&&p.length===2?[number(p[0],0,width),number(p[1],0,height)]:fail())};
  });
}

async function analysis(value,width,height,inputHash){
  if(value==null)return null;
  object(value);
  if(value.width!==width||value.height!==height||sha(value.input_sha256)!==inputHash)fail();
  const mask=bytesFromBase64(value.mask_png_base64,20*MB),header=new DataView(mask.buffer,mask.byteOffset,mask.byteLength);
  if(mask.length<33||[137,80,78,71,13,10,26,10].some((b,i)=>mask[i]!==b)||header.getUint32(8)!==13||
    String.fromCharCode(...mask.slice(12,16))!=='IHDR'||header.getUint32(16)!==width||header.getUint32(20)!==height||mask[24]!==8||mask[25]!==0)fail();
  const maskHash=await hash(mask);
  if(value.mask_sha256!==undefined&&sha(value.mask_sha256)!==maskHash)fail();
  if(typeof createImageBitmap==='function'){
    const bitmap=await createImageBitmap(new Blob([mask],{type:'image/png'}));
    try{if(bitmap.width!==width||bitmap.height!==height)fail();}finally{bitmap.close();}
  }
  const n=integer(value.n_regions,0,width*height);
  if(!Array.isArray(value.regions)||value.regions.length>32||value.regions.length>n)fail();
  const regions=value.regions.map((r,i)=>{
    object(r);const box=rectangle(r,width,height),area=integer(r.area_px,1,box.width*box.height),fraction=number(r.area_frac,0,1);
    if(Math.abs(fraction-area/(width*height))>1e-6||r.id!==undefined&&r.id!==i+1)fail();
    return {...box,area_px:area,area_frac:fraction,mean_prob:number(r.mean_prob,0,1),max_prob:number(r.max_prob,0,1)};
  });
  if(typeof value.manipulated!=='boolean')fail();
  const result=modelResult({width,height,input_sha256:inputHash,mask_png_base64:value.mask_png_base64,mask_sha256:maskHash,
    prediction_id:text(value.prediction_id,128),analysed_at:date(value.analysed_at),model:model(value.model),
    manipulated:value.manipulated,score:number(value.score,-1e6,1e6),area_frac:number(value.area_frac,0,1),
    n_regions:n,regions,inference_ms:number(value.inference_ms,0,1e9)});
  for(const key of ['score_kind','score_calibrated'])if(value[key]!==undefined){
    result[key]=key==='score_kind'?text(value[key]):boolean(value[key]);
    if(result.model[key]!==undefined&&result.model[key]!==result[key])fail();
  }
  return result;
}

function explanations(values,prediction,width,height){
  object(values);const result={};
  if(Object.keys(values).length>32)fail();
  for(const [key,value] of Object.entries(values)){
    object(value);const regionId=integer(value.region_id,1,32),d=object(value.description);
    if(String(regionId)!==key||!prediction||!prediction.regions.some(r=>r.id===regionId)||value.source!=='openrouter_vlm'||
      value.prediction_id!==prediction.prediction_id||value.input_sha256!==prediction.input_sha256||
      value.mask_sha256!==prediction.mask_sha256||JSON.stringify(model(value.detector_model))!==JSON.stringify(prediction.model)||
      !Object.hasOwn(operationLabels,d.operation)||!Array.isArray(d.evidence)||d.evidence.length>20)fail();
    const inputs=object(value.inputs);
    for(const key of ['image_sizes','views'])if(inputs[key]!==undefined&&(!Array.isArray(inputs[key])||inputs[key].length>3))fail();
    if(inputs.image_sizes&&inputs.views&&inputs.image_sizes.length!==inputs.views.length)fail();
    let materials;
    if(inputs.selected_region!==undefined){
      const selected=object(inputs.selected_region),region=prediction.regions.find(r=>r.id===regionId);
      const fields=['x','y','width','height','area_px','area_frac','mean_prob','max_prob'];
      if(fields.some(k=>selected[k]!==region[k])||inputs.mask_scope!=='selected_component'||
        JSON.stringify(inputs.views)!==JSON.stringify(['image','mask','overlay'])||inputs.image_sizes?.length!==3)fail();
      const sizes=inputs.image_sizes.map(dimensions);
      if(sizes.some(s=>s.width!==sizes[0].width||s.height!==sizes[0].height))fail();
      materials={selected_region:Object.fromEntries(fields.map(k=>[k,selected[k]])),mask_scope:'selected_component',image_sizes:sizes,views:inputs.views};
    }else materials={crop:rectangle(object(inputs.crop),width,height),
      ...(inputs.image_sizes?{image_sizes:inputs.image_sizes.map(dimensions)}:{}),
      ...(inputs.views?{views:inputs.views.map(v=>['full_image','clean_crop','outlined_crop'].includes(v)?v:fail())}:{})};
    result[key]={region_id:regionId,source:'openrouter_vlm',prediction_id:prediction.prediction_id,
      input_sha256:prediction.input_sha256,mask_sha256:prediction.mask_sha256,detector_model:prediction.model,
      description:{object:text(d.object,5000),observation:text(d.observation,5000),possible_change:text(d.possible_change,5000),
        operation:d.operation,evidence:d.evidence.map(s=>text(s,1000)),limitations:text(d.limitations,5000)},
      model:text(value.model),generated_at:date(value.generated_at),prompt_version:text(value.prompt_version),
      elapsed_ms:number(value.elapsed_ms,0,1e9),limitation:text(value.limitation,5000),
      inputs:materials};
    for(const name of ['requested_model','requested_provider','provider','request_id'])if(value[name]!==undefined)result[key][name]=optionalText(value[name]);
    if(value.cached!==undefined)result[key].cached=boolean(value.cached);
    if(value.usage!==undefined)result[key].usage=Object.fromEntries(Object.entries(object(value.usage))
      .filter(([k])=>['prompt_tokens','completion_tokens','total_tokens','cost'].includes(k))
      .map(([k,v])=>[k,v===null?null:number(v,0,1e12)]));
  }
  return result;
}

async function restoreProject(value,child=false){
  object(value);
  if(value.format!==FORMAT||value.format_version!==PROJECT_FORMAT_VERSION)fail();
  const saved_at=date(value.saved_at),item=object(value.item),{width,height}=dimensions(item);
  const kind=item.kind;
  if(item.source!=='upload'||!['image','video'].includes(kind)||child&&(kind!=='image'||item.videoFrames!==undefined||item.videoScan!==undefined))fail();
  const filename=text(item.filename,500),id=text(item.id,128);
  if(!id||!filename)fail();
  const {file,inputHash}=await media(item.original,kind,filename,width,height);
  const prediction=await analysis(item.analysis,width,height,inputHash);
  if(kind==='video'&&prediction)fail();
  const result={id,filename,kind,source:'upload',width,height,file,analysis:prediction,
    regions:manualRegions(item.regions,width,height,prediction?.regions.map(r=>r.id)||[]),
    explanations:explanations(item.explanations,prediction,width,height),reviewDraft:review(item.review),saved_at,restored:true};
  if(kind==='video'){
    result.duration=number(item.duration,.001,60);
    result.video_time=number(item.video_time??0,0,result.duration);
  }
  if(item.captured_at!=null)result.captured_at=number(item.captured_at,0,60);
  if(item.parentVideoId!==undefined){result.parentVideoId=text(item.parentVideoId,128);if(!result.parentVideoId)fail();}
  if(item.reference!==undefined){
    if(kind!=='image')fail();
    const reference=object(item.reference),size=dimensions(reference),name=text(reference.filename,500);
    if(!name)fail();
    result.reference={file:(await media(reference.original,'image',name,size.width,size.height)).file,...size};
  }
  if(item.videoFrames!==undefined||item.videoScan!==undefined){
    if(kind!=='video'||!Array.isArray(item.videoFrames)||item.videoFrames.length>12)fail();
    if(item.videoFrames.reduce((sum,p)=>sum+encodedSize(object(object(p).item).original?.base64),0)>50*MB)fail();
    const frames=[];const ids=new Set([id]);
    for(const project of item.videoFrames){
      const frame=await restoreProject(project,true);
      if(ids.has(frame.id)||frame.parentVideoId!==id||frame.captured_at===undefined||frame.captured_at>=result.duration)fail();
      ids.add(frame.id);frame.originalFile=frame.file;frames.push(frame);
    }
    result.videoFrames=frames;result.videoScan=scan(item.videoScan,result.duration,frames);
  }
  return result;
}

export const deserializeProject=value=>restoreProject(value);

async function encodeSnapshot(item){
  const original=await encodeFile(item.file,item.kind);
  const project={format:FORMAT,format_version:PROJECT_FORMAT_VERSION,saved_at:new Date().toISOString(),
    item:{id:item.id,filename:item.filename,kind:item.kind,source:item.source,width:item.width,height:item.height,
      original,analysis:item.analysis,regions:item.regions,explanations:item.explanations,review:item.review}};
  for(const key of ['duration','video_time','captured_at','parentVideoId','videoScan'])if(item[key]!=null)project.item[key]=item[key];
  if(item.reference)project.item.reference={filename:item.reference.file.name,width:item.reference.width,height:item.reference.height,
    original:await encodeFile(item.reference.file,'image')};
  if(item.videoFrames)project.item.videoFrames=await Promise.all(item.videoFrames.map(encodeSnapshot));
  return project;
}

export async function serializeProject(item,operatorReview=item.reviewDraft||{decision:'pending',note:''}){
  const project=await encodeSnapshot(snapshot(item,operatorReview));
  return canonicalProject(project,await deserializeProject(project));
}

function canonicalProject(project,item){
  project.item.analysis=item.analysis;project.item.regions=item.regions;project.item.explanations=item.explanations;project.item.review=item.reviewDraft;
  if(item.videoScan)project.item.videoScan=item.videoScan;
  if(item.videoFrames)project.item.videoFrames=project.item.videoFrames.map((p,i)=>canonicalProject(p,item.videoFrames[i]));
  return project;
}

function encodedSize(value){
  if(typeof value!=='string'||value.length%4)fail();
  return value.length/4*3-(value.endsWith('==')?2:value.endsWith('=')?1:0);
}

export function projectBinarySize(project){
  const item=object(object(project).item);
  let size=encodedSize(object(item.original).base64);
  if(item.reference)size+=encodedSize(object(item.reference.original).base64);
  if(item.analysis)size+=encodedSize(item.analysis.mask_png_base64);
  if(item.videoFrames!==undefined){
    if(!Array.isArray(item.videoFrames)||item.videoFrames.length>12)fail();
    for(const frame of item.videoFrames){if(object(object(frame).item).videoFrames!==undefined)fail();size+=projectBinarySize(frame);}
  }
  return size;
}

export function archiveDatabase(dbName,storeName,action,mode='readonly'){
  return new Promise((resolve,reject)=>{
    if(!globalThis.indexedDB){reject(new Error('Браузер не поддерживает локальное хранение проверок. Сохраните файл проекта.'));return;}
    const request=indexedDB.open(dbName,1);let abandoned=false;
    request.onupgradeneeded=()=>request.result.createObjectStore(storeName,{keyPath:'id'});
    request.onerror=()=>{abandoned=true;reject(request.error);};
    request.onblocked=()=>{abandoned=true;reject(new Error('Закройте другие вкладки сервиса и повторите сохранение.'));};
    request.onsuccess=()=>{
      const db=request.result;
      if(abandoned){db.close();return;}
      db.onversionchange=()=>db.close();
      const transaction=db.transaction(storeName,mode);let operation;
      try{operation=action(transaction.objectStore(storeName));}catch(error){db.close();reject(error);return;}
      transaction.oncomplete=()=>{db.close();resolve(operation.result);};
      transaction.onerror=transaction.onabort=()=>{db.close();reject(transaction.error||operation.error||new Error('Не удалось сохранить проверку.'));};
    };
  });
}

const database=(action,mode)=>archiveDatabase(DB,STORE,action,mode);

export async function saveReview(project){
  const input=structuredClone(project),item=await deserializeProject(input),summary={id:item.id,filename:item.filename,source:'upload',saved_at:item.saved_at,
    review:item.reviewDraft,region_count:(item.analysis?.n_regions||0)+item.regions.length,kind:item.kind};
  const stored=canonicalProject(input,item);delete stored.item.original.base64;
  await database(store=>store.put({id:item.id,summary,project:stored,original_blob:item.file}),'readwrite');
  return summary;
}

export async function listReviews(){
  const rows=await database(store=>store.getAll());
  return rows.map(row=>row.summary).sort((a,b)=>b.saved_at.localeCompare(a.saved_at));
}

export async function loadReview(id){
  const row=await database(store=>store.get(id));
  if(!row)return null;
  row.project.item.original.base64=await base64(row.original_blob);
  return deserializeProject(row.project);
}

export const deleteReview=id=>database(store=>store.delete(id),'readwrite');
export const clearReviews=()=>database(store=>store.clear(),'readwrite');
