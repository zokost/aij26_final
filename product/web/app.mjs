import {fixtures,frameRegions,statusFor,reportData,modelResult,operationLabels} from './data.mjs?v=features-20261019c';
import {serializeProject,deserializeProject,saveReview,listReviews,loadReview,deleteReview,clearReviews} from './review-store.mjs?v=features-20261019c';
import {fitScale,imagePoint,dividerPercent,nearDivider,pointInPolygon} from './viewer.mjs';
import {analyseVideo} from './video-analysis.mjs?v=features-20261019c';
import {mountReferenceComparison} from './reference-compare.mjs?v=features-20261019c';
import {serializeCase,deserializeCase,saveCase,listCases,loadCase,deleteCase,caseSummary} from './case-store.mjs?v=features-20261019c';

const $=id=>document.getElementById(id);
const icon=name=>`<svg class="icon" aria-hidden="true"><use href="#i-${name}"/></svg>`;
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const percent=n=>n>0&&n*100<.005?'<0,01%':(n*100).toLocaleString('ru-RU',{maximumFractionDigits:2,minimumFractionDigits:n>0&&n<.01?2:0})+'%';
const time=n=>`${Math.floor(n/60).toString().padStart(2,'0')}:${Math.floor(n%60).toString().padStart(2,'0')}`;
const regionWord=n=>n%10===1&&n%100!==11?'область':n%10>=2&&n%10<=4&&!(n%100>=12&&n%100<=14)?'области':'областей';
const canvas=$('main-canvas'),ctx=canvas.getContext('2d');
const uploaded=[];
let current=null,selected=null,second=7,view='overlay',zoom=1,drawStart=null,draft=null,marking=false,loadToken=0,playTimer=null,toastTimer=null;
let gesture=null,renderScale=1,viewerFit=1,viewerNeedsLayout=true,resetScroll=true,observedViewerWidth=0,observedViewerHeight=0;
let live=false,knownModel=false,device='',requestController=null,loadingItem=null,importing=false,batchGeneration=0;
let batchProgress=null,healthBusy=false,healthTimer=null,projectBusy=false,restoreToken=0;
let explanationService={available:false,model:'deepseek/deepseek-v4.1-flash'},explanationBusy=false,explanationController=null,explanationItem=null;
let saved=[],legacySaved=[];
let activeCase={id:crypto.randomUUID(),title:'Новое обращение',note:'',decision:'pending'},caseSaved=[],caseBusy=false,videoJob=null,reportSnapshot=null,caseRevision=0;
let activePage='history';
// Keep the existing controls and their state while giving each workflow its own page.
$('case-body').append(document.querySelector('.case-card'),$('drop-zone'),$('queue'));
$('archive-tools').append($('open-case'),$('open-project'));
$('guide').insertBefore(document.querySelector('.scenario-row'),$('guide').querySelector('.guide-grid'));
$('explanation-panel').append(document.querySelector('.vlm-card'));
$('integration-api').textContent=location.origin;
const mcpOrigin=['localhost','127.0.0.1','[::1]'].includes(location.hostname)?location.origin:'http://127.0.0.1:8765';
$('integration-mcp').textContent=`python -m product.mcp_server --image-root . --api-url ${mcpOrigin}`;
try {const value=JSON.parse(localStorage.getItem('detective-demo-history')||'[]');if(Array.isArray(value))legacySaved=value.filter(e=>e&&typeof e.id==='string'&&typeof e.filename==='string').slice(0,30);} catch { /* Storage may be unavailable in private browser sessions. */ }
const reviewLabels={pending:'Не определено',review:'Передать на проверку',accepted:'Без замечаний оператора'};
const caseStatusLabels={empty:'Нет материалов',in_progress:'Есть незавершённые проверки',needs_review:'Нужен просмотр оператора',reviewed:'Решения оператора записаны'};
const scanLabels={complete:'Выборка проверена',cancelled:'Проверка отменена',error:'Проверка завершилась ошибкой',running:'Проверка кадров'};
const regions=()=>current?frameRegions(current,second):[];
const activeReview=()=>({decision:$('review-select').value,note:$('review-note').value.trim()});

function toast(message) {clearTimeout(toastTimer);$('toast').textContent=message;$('toast').hidden=false;toastTimer=setTimeout(()=>$('toast').hidden=true,4200);}
const referenceComparison=mountReferenceComparison({host:$('reference-host'),getCurrent:()=>current,onChange:()=>{$('save-status').textContent='Есть несохранённые изменения';caseChanged();},toast});
const videoOwner=(item=current)=>item?.kind==='video'?item:uploaded.find(video=>video.id===item?.parentVideoId&&video.kind==='video');
const frameTime=n=>`${time(n)}.${Math.floor(n%1*100).toString().padStart(2,'0')}`;
function stopPlayback(){clearInterval(playTimer);playTimer=null;$('play-frames').textContent='Воспроизвести';}
function loadImage(src,signal){return new Promise((resolve,reject)=>{const image=new Image();const finish=(error)=>{signal?.removeEventListener('abort',abort);image.onload=image.onerror=null;error?reject(error):resolve(image);};const abort=()=>{image.src='';finish(new DOMException('Ожидание отменено','AbortError'));};image.onload=()=>finish();image.onerror=()=>finish(new Error('Не удалось открыть изображение'));if(signal?.aborted){abort();return;}signal?.addEventListener('abort',abort,{once:true});image.src=src;});}

async function analyse(item,signal){
  const body=new FormData();body.append('file',item.file);
  const response=await fetch('/detect?mask=true',{method:'POST',body,signal});
  const data=await response.json().catch(()=>({detail:'Сервис модели не ответил. Проверьте, что он запущен.'}));
  signal.throwIfAborted();
  if(!response.ok)throw new Error(data.detail||'Не удалось выполнить проверку');
  if(data.width!==item.width||data.height!==item.height)throw new Error('Размер ответа не совпадает с изображением. Проверьте ориентацию файла.');
  const mask=await loadImage('data:image/png;base64,'+data.mask_png_base64,signal);
  if(mask.naturalWidth!==item.width||mask.naturalHeight!==item.height)throw new Error('Размер маски не совпадает с изображением');
  signal.throwIfAborted();if(item.removed)throw new DOMException('Материал удалён','AbortError');
  item.analysis=modelResult(data);item.maskImage=mask;
}

function colorMask(item){
  if(item.colorMask)return item.colorMask;
  const c=document.createElement('canvas');c.width=item.width;c.height=item.height;
  const cctx=c.getContext('2d');cctx.drawImage(item.maskImage,0,0);
  const pixels=cctx.getImageData(0,0,c.width,c.height),d=pixels.data;
  for(let i=0;i<d.length;i+=4){const on=d[i]>=128;d[i]=225;d[i+1]=159;d[i+2]=65;d[i+3]=on?255:0;}
  cctx.putImageData(pixels,0,0);item.colorMask=c;return c;
}

fixtures.forEach(f=>{
  const button=document.createElement('button');button.className='scenario-card';button.dataset.fixture=f.id;
  button.innerHTML=`<img src="${f.src}" alt=""><span><strong>${f.name}</strong><small>${f.subtitle}</small></span><span class="scenario-check" hidden>${icon('check')}</span>`;
  button.addEventListener('click',()=>openItem({...f,source:'fixture',regions:structuredClone(f.regions)}));$('scenario-cards').append(button);
});

async function openItem(item,{batch=false,navigate=true}={}){
  if(caseBusy||videoJob){toast('Дождитесь завершения операции или отмените ожидание видео');return false;}
  if(importing&&!batch){toast('Дождитесь обработки очереди или отмените ожидание');return false;}
  if(item.removed)return;
  ++restoreToken;
  if(current){current.reviewDraft=activeReview();if(current.kind==='video'&&$('uploaded-video').src===current.src)current.video_time=$('uploaded-video').currentTime;}
  const token=++loadToken;stopPlayback();cancelGesture();marking=false;zoom=1;
  requestController?.abort();requestController=new AbortController();
  const signal=requestController.signal;loadingItem=item;item.cancelled=false;item.queued=false;
  $('upload-empty').hidden=true;$('report-button').disabled=true;$('mask-download').disabled=true;
  $('save-project').disabled=true;['save-review','review-select','review-note'].forEach(id=>$(id).disabled=true);
  $('loading-title').textContent='Подготовка просмотра';
  $('loading-note').textContent='Файл открывается на вашем компьютере';
  $('loading').hidden=false;$('uploaded-video').pause();$('uploaded-video').hidden=true;$('canvas-scroll').hidden=false;
  renderQueue();
  try {
    if(item.file&&!item.src)item.src=URL.createObjectURL(item.file);
    if(item.kind==='video'){
      const video=$('uploaded-video');video.src=item.src;
      await new Promise((resolve,reject)=>{const abort=()=>finish(new DOMException('Ожидание отменено','AbortError'));const finish=error=>{signal.removeEventListener('abort',abort);video.onloadedmetadata=video.onerror=null;error?reject(error):resolve();};video.onloadedmetadata=()=>finish();video.onerror=()=>finish(new Error('Видеоформат не поддерживается браузером'));signal.addEventListener('abort',abort,{once:true});if(signal.aborted)abort();});
      if(video.duration>60||!Number.isFinite(video.duration))throw new Error('Для прототипа выберите видео длительностью до 60 секунд');
      if(video.videoWidth*video.videoHeight>15_000_000)throw new Error('Разрешение видео превышает 15 млн пикселей');
      if(item.restored&&(video.videoWidth!==item.width||video.videoHeight!==item.height||Math.abs(video.duration-item.duration)>.05))throw new Error('Параметры сохранённого видео не совпадают с файлом');
      item.width=video.videoWidth;item.height=video.videoHeight;item.duration=video.duration;item.element=video;
      if(item.video_time)video.currentTime=Math.min(item.video_time,video.duration);
    } else {item.element=await loadImage(item.src,signal);const width=item.element.naturalWidth,height=item.element.naturalHeight;if(item.restored&&(width!==item.width||height!==item.height))throw new Error('Размер сохранённого изображения не совпадает с файлом');item.width=width;item.height=height;if(item.width*item.height>15_000_000)throw new Error('Изображение превышает 15 млн пикселей');}
    signal.throwIfAborted();if(token!==loadToken||item.removed)return;
    if(live&&!item.restored&&item.source==='upload'&&item.kind==='image'&&!item.analysis){
      item.pending=true;renderQueue();$('loading-title').textContent='Проверка моделью';
      $('loading-note').textContent=`Обработка на ${device||'локальном устройстве'}. Первый запрос может занять несколько секунд.`;
      await analyse(item,signal);
    }else if(item.analysis&&!item.maskImage){item.maskImage=await loadImage('data:image/png;base64,'+item.analysis.mask_png_base64,signal);if(item.maskImage.naturalWidth!==item.width||item.maskImage.naturalHeight!==item.height)throw new Error('Размер сохранённой маски не совпадает с изображением');}
    signal.throwIfAborted();if(token!==loadToken||item.removed)return;
    if(current&&current!==item&&current.kind==='image'){current.element=null;current.maskImage=null;current.colorMask=null;}
    item.error=null;item.queued=false;current=item;viewerNeedsLayout=true;resetScroll=true;second=item.kind==='storyboard'?7:0;selected=regions()[0]?.id??null;
    $('uploaded-video').hidden=item.kind!=='video';$('canvas-scroll').hidden=item.kind==='video';
    const entry=saved.find(e=>e.id===item.id),review=item.reviewDraft||entry?.review;
    $('review-select').value=review?.decision in reviewLabels?review.decision:'pending';$('review-note').value=review?.note||'';
    $('save-status').textContent=item.restored?'Сохранённая проверка · новый анализ не запускался':entry?'Проверка сохранена':'Сохраните проверку, чтобы открыть её позже';
    if(navigate)showPage('workspace');render();
    if(!batch&&navigate&&item.source==='upload')$('filename').closest('.viewer-card').scrollIntoView({block:'start'});
    return true;
  }catch(error){if(item!==current){item.element=null;item.maskImage=null;item.colorMask=null;}item.queued=false;if(error.name==='AbortError')item.cancelled=true;else item.error=error.message;if(token===loadToken){if(error.name!=='AbortError')toast(error.message);if(current&&(current.kind!=='video'||$('uploaded-video').src===current.src))render();else renderEmpty();}return false;}
  finally{item.pending=false;if(token===loadToken){loadingItem=null;$('loading').hidden=true;$('report-button').disabled=!current;$('mask-download').disabled=!current;updateSaveControls();renderQueue();}}
}

function render(){
  if(!current)return;
  $('uploaded-video').hidden=current.kind!=='video';$('canvas-scroll').hidden=current.kind==='video';
  const list=regions(),status=current.kind==='video'?videoStatus(current):statusFor(current,list);
  ['save-review','review-select','review-note'].forEach(id=>$(id).disabled=false);
  updateSaveControls();
  $('upload-empty').hidden=true;
  $('filename').textContent=current.filename;
  $('filemeta').textContent=current.kind==='storyboard'?`Сценарий видео · кадр ${time(second)} / ${time(current.duration)}`:current.kind==='video'?`Локальное видео · ${time(current.duration)} · проверено кадров: ${current.videoFrames?.length||0}`:`${current.width} × ${current.height} px · ${current.captured_at!=null?'кадр '+frameTime(current.captured_at):current.source==='fixture'?'иллюстрация':'локальный файл'}`;
  $('source-tag').textContent=current.restored?'Сохранённая проверка':current.analysis?'Результат модели':current.source==='fixture'?'Учебный пример':'Ваш файл';
  $('stage-label').textContent=current.analysis?'Маска модели · порог '+current.analysis.model.threshold:current.source==='fixture'?'Демонстрационная разметка':list.length?'Ручные отметки':'Просмотр · без анализа';
  $('stage-label').parentElement.hidden=current.kind==='video';
  $('canvas-hint').textContent=marking?'Нажмите и протяните прямоугольник для ручной отметки':list.length?'Выберите область на изображении, чтобы рассмотреть её':current.source==='fixture'?'В этом учебном кадре области не выделены':'Можно добавить ручную отметку с помощью инструмента в панели';
  $('canvas-hint').hidden=current.kind==='video';
  $('image-stage').classList.toggle('has-image',current.kind!=='video');
  $('mark-button').disabled=current.kind==='video'||current.source!=='upload';$('mark-button').classList.toggle('active',marking);
  canvas.classList.toggle('marking',marking);
  if(current.source==='upload')document.body.classList.remove('demo-examples');
  $('dimensions').textContent=`${current.width} × ${current.height} px`;
  $('status-title').textContent=status.label;$('status-detail').textContent=status.detail;$('status-icon').className=`status-icon ${status.tone}`;
  const total=current.kind==='video'?(current.videoFrames||[]).reduce((n,f)=>n+(f.analysis?.n_regions||0)+f.regions.length,0):current.analysis?current.analysis.n_regions+current.regions.length:list.length;
  $('regions-count').textContent=total;$('findings-count').textContent=total;
  $('regions-word').textContent=regionWord(total);
  // shortcut: manual areas are summed; use polygon union if their overlap matters.
  $('area-value').textContent=current.analysis?percent(current.analysis.area_frac):list.length?percent(list.reduce((n,r)=>n+r.area_frac,0)):'—';
  $('area-value').title=current.analysis?'Доля пикселей в точной маске модели; ручные отметки не включены':'Сумма площадей областей; пересечения учитываются повторно';
  $('area-value').nextElementSibling.textContent=current.analysis?'маска модели':'сумма площадей';
  $('result-number').textContent=current.source==='fixture'?String(fixtures.findIndex(f=>f.id===current.id)+1).padStart(2,'0'):'LOCAL';
  $('viewer-note').textContent=current.kind==='video'?'Откройте кадр на временной шкале для просмотра маски.':current.analysis?`Модель: ${current.analysis.inference_ms.toFixed(0)} мс · ручные отметки показаны синим.`:current.source==='fixture'?'Учебная маска показывает область внимания, а не причину изменения.':'Ручная разметка не является автоматическим результатом проверки.';
  $('mask-download').innerHTML=icon('download')+(current.analysis?'Маска модели':'PNG-маска');
  $('timeline').hidden=current.kind!=='storyboard';$('video-tools').hidden=current.kind!=='video';
  document.querySelectorAll('.scenario-card').forEach(b=>{const active=b.dataset.fixture===current.id;b.classList.toggle('active',active);b.querySelector('.scenario-check').hidden=!active;b.setAttribute('aria-pressed',active);});
  if(current.kind==='storyboard')renderTimeline();
  renderRegions();renderMetadata();renderQueue();draw();renderVideoScan();referenceComparison.render();
}

function path(context,points){context.beginPath();points.forEach(([x,y],i)=>i?context.lineTo(x,y):context.moveTo(x,y));context.closePath();}
function drawBase(context){context.drawImage(current.element,0,0,current.width,current.height);}
function drawMask(context,list,alpha=.32,labels=true,strokeWidth=Math.max(2,current.width/550)){
  context.save();context.lineWidth=strokeWidth;
  if(current.analysis){context.save();context.globalAlpha=alpha;context.drawImage(colorMask(current),0,0);context.restore();}
  list.forEach(r=>{
    if(r.polygon){path(context,r.polygon);context.fillStyle=r.manual?`rgba(83,151,198,${alpha})`:`rgba(225,159,65,${alpha})`;context.fill();}
    else{context.beginPath();context.rect(r.x,r.y,r.width,r.height);}
    context.strokeStyle=r.manual?'#76b5df':r.id===selected?'#f1d896':'#d9b971';context.stroke();
    if(labels){context.fillStyle=r.id===selected?'#f6eccb':'#264c37';context.fillRect(r.x,r.y-29,29,26);context.font='600 15px Segoe UI, Arial';context.textAlign='center';context.fillStyle=r.id===selected?'#405333':'#e7e9bd';context.fillText(String(r.id),r.x+14,r.y-11);}
  });context.restore();
}
function draw(){
  if(!current||current.kind==='video')return;
  const scroll=$('canvas-scroll'),bounds=scroll.getBoundingClientRect();if(!bounds.width||!bounds.height)return;
  if(viewerNeedsLayout){viewerFit=fitScale(current.width,current.height,bounds.width,bounds.height);viewerNeedsLayout=false;}
  renderScale=viewerFit*zoom;
  canvas.width=current.width;canvas.height=current.height;canvas.style.width=`${current.width*renderScale}px`;canvas.style.height=`${current.height*renderScale}px`;
  $('canvas-content').style.width=canvas.style.width;$('canvas-content').style.height=canvas.style.height;
  $('zoom-reset').textContent=zoom===1?'Вписано':`${Math.round(zoom*100)}%`;
  $('zoom-reset').title=`Вписать целиком · ${Math.round(viewerFit*100)}% исходного размера`;
  const dragger=$('compare-dragger');dragger.hidden=view!=='compare'||marking;dragger.style.left=$('compare-slider').value+'%';
  if(resetScroll){scroll.scrollTop=0;scroll.scrollLeft=0;resetScroll=false;}
  drawBase(ctx);
  if(view!=='original'){
    ctx.save();const split=current.width*Number($('compare-slider').value)/100;
    if(view==='compare'){ctx.beginPath();ctx.rect(split,0,current.width-split,current.height);ctx.clip();}
    drawMask(ctx,regions(),Number($('opacity').value)/100);
    ctx.restore();
    if(view==='compare'){ctx.strokeStyle='#f2f0d8';ctx.lineWidth=2/renderScale;ctx.beginPath();ctx.moveTo(split,0);ctx.lineTo(split,current.height);ctx.stroke();ctx.fillStyle='#f2f0d8';ctx.beginPath();ctx.arc(split,current.height/2,17/renderScale,0,Math.PI*2);ctx.fill();ctx.fillStyle='#526843';ctx.font=`${19/renderScale}px Arial`;ctx.textAlign='center';ctx.fillText('↔',split,current.height/2+6/renderScale);}
  }
  if(draft){ctx.strokeStyle='#f3e3ab';ctx.lineWidth=3;ctx.setLineDash([8,5]);ctx.strokeRect(draft.x,draft.y,draft.w,draft.h);ctx.setLineDash([]);}
  drawCrop();
}

function renderRegions(){
  const list=regions();if(!list.some(r=>r.id===selected))selected=list[0]?.id??null;
  $('region-list').replaceChildren();list.forEach(r=>{
    const button=document.createElement('button');button.className=`region-button ${r.id===selected?'active':''}`;button.setAttribute('aria-pressed',r.id===selected);
    button.innerHTML=`<span class="region-number">${r.id}</span><span><strong>${esc(r.title)}</strong><small>${r.manual?'Ручная · ':''}${percent(r.area_frac)} кадра · ${Math.round(r.width)} × ${Math.round(r.height)} px</small></span>${icon('arrow')}`;
    button.addEventListener('click',()=>{selected=r.id;renderRegions();draw();});$('region-list').append(button);
  });
  $('no-regions').hidden=!!list.length;$('region-detail').hidden=!list.length;
  $('no-regions').querySelector('strong').textContent=current.kind==='video'?'Откройте находки в кадрах':'Нет выделенных областей';
  $('empty-detail').textContent=current.kind==='video'?'Откройте проверенный кадр на временной шкале, чтобы рассмотреть его маску и записать решение.':current.analysis?'Модель вернула пустую маску при текущем пороге. Это не подтверждает подлинность.':current.source==='fixture'?'В этом учебном примере маска пуста. Подлинность материала из этого не следует.':'Автоматического результата нет. Можно добавить ручную отметку.';
  const omitted=current.analysis&&current.analysis.n_regions>current.analysis.regions.length;
  $('region-list-note').hidden=!omitted;
  if(omitted)$('region-list-note').textContent=`Показаны ${current.analysis.regions.length} из ${current.analysis.n_regions} областей модели. PNG-маска содержит все области.`;
  const region=list.find(r=>r.id===selected);if(region){$('explanation').textContent=region.hint;$('crop-scale').textContent=`Область ${region.id}`;}
  $('remove-region').hidden=!(region?.manual);
  renderExplanation();
}
function renderExplanation(){
  const region=regions().find(r=>r.id===selected),result=current?.explanations?.[selected];
  const eligible=region?.annotation_source==='model'&&current?.analysis&&current?.file&&!current.restored;
  $('vlm-button').disabled=!eligible||!explanationService.available||explanationBusy||importing;
  $('vlm-button').textContent=result?'Пояснение сохранено':current?.explanationPending===selected?'Запрашиваем пояснение…':'Объяснить область';
  if(result)$('vlm-button').disabled=true;
  $('vlm-badge').textContent=explanationService.available?(explanationService.free?'Бесплатный тест':'По запросу'):'Не подключено';
  $('vlm-privacy').textContent='По нажатию в OpenRouter и Together отправляются фото, бинарная маска выбранной области и фото с жёлтым наложением. '+(explanationService.free?'Бесплатная модель; действует квота запросов.':'Запрос платный.');
  $('vlm-status').textContent=current?.explanationPending===selected?'Запрос в OpenRouter. Можно продолжать просмотр; маска не меняется.':
    result?'Гипотеза VLM · область '+selected:current?.explanationErrors?.[selected]||
    (current?.restored?'Сохранённая проверка. Доступны уже полученные пояснения; новые внешние запросы не выполняются.':!eligible?'Выберите область реального результата модели.':!explanationService.available?explanationService.reason||'Внешние пояснения выключены на сервере.':explanationBusy?'Дождитесь текущего пояснения.':'Пояснение ещё не запрашивалось.');
  const d=result?.description;
  for(const key of ['object','observation','possible_change','limitations'])$('vlm-'+key).textContent=d?.[key]||'—';
  $('vlm-operation').textContent=d?operationLabels[d.operation]||operationLabels.unknown:'—';
  $('vlm-evidence').textContent=d?d.evidence.join(' · ')||'Видимых оснований недостаточно':'—';
  $('vlm-meta').textContent=result?`${result.model} · ${new Date(result.generated_at).toLocaleString('ru-RU')} · ${(result.elapsed_ms/1000).toFixed(1)} с`:explanationService.model;
}
$('vlm-button').addEventListener('click',async()=>{
  const item=current,region=regions().find(r=>r.id===selected);
  if(explanationBusy||!explanationService.available||region?.annotation_source!=='model'||!item?.file||item.restored||item.explanations?.[region.id])return;
  explanationBusy=true;explanationItem=item;explanationController=new AbortController();const signal=explanationController.signal;item.explanationPending=region.id;renderExplanation();renderQueue();$('save-project').disabled=true;$('save-review').disabled=true;
  try{
    const png=Uint8Array.from(atob(item.analysis.mask_png_base64),c=>c.charCodeAt(0));
    const body=new FormData();body.append('file',item.file);body.append('mask',new Blob([png],{type:'image/png'}),'mask.png');
    body.append('prediction_id',item.analysis.prediction_id);body.append('region_id',region.id);
    const response=await fetch('/explain',{method:'POST',body,signal});
    const result=await response.json().catch(()=>({detail:'Сервис пояснений не ответил.'}));
    signal.throwIfAborted();
    if(!response.ok)throw new Error(result.detail||'Не удалось получить пояснение.');
    signal.throwIfAborted();if(item.removed)return;
    if(result.prediction_id!==item.analysis.prediction_id||result.region_id!==region.id||result.input_sha256!==item.analysis.input_sha256)throw new Error('Пояснение не соответствует выбранному материалу.');
    item.explanations??={};item.explanations[region.id]=result;
    if(item.explanationErrors)delete item.explanationErrors[region.id];
    if(current===item)$('save-status').textContent='Пояснение получено. Сохраните решение или экспортируйте отчёт.';
  }catch(error){if(!item.removed){item.explanationErrors??={};item.explanationErrors[region.id]=error.name==='AbortError'?'Ожидание пояснения отменено. Провайдер мог завершить запрос.':error.message;if(current===item)toast(item.explanationErrors[region.id]);}}
  finally{explanationBusy=false;explanationItem=null;explanationController=null;item.explanationPending=null;renderExplanation();renderQueue();updateSaveControls();}
});
function drawCrop(crop=$('crop-canvas'),width=500,height=240,showMask=view!=='original',zoomFactor=1){
  const r=regions().find(r=>r.id===selected);if(!r||current.kind==='video')return;
  const c=crop.getContext('2d');
  const padding=Math.max(22,Math.max(r.width,r.height)*.28);
  const w=Math.min(current.width,r.width+padding*2)/zoomFactor,h=Math.min(current.height,r.height+padding*2)/zoomFactor;
  const x=Math.max(0,Math.min(current.width-w,r.x+r.width/2-w/2)),y=Math.max(0,Math.min(current.height-h,r.y+r.height/2-h/2));
  crop.width=width;crop.height=height;c.fillStyle='#dce5d5';c.fillRect(0,0,width,height);
  const scale=Math.min(width/w,height/h),dx=(width-w*scale)/2,dy=(height-h*scale)/2;
  c.drawImage(current.element,x,y,w,h,dx,dy,w*scale,h*scale);
  if(showMask){c.save();c.beginPath();c.rect(dx,dy,w*scale,h*scale);c.clip();c.translate(dx-x*scale,dy-y*scale);c.scale(scale,scale);drawMask(c,[r],Number($('opacity').value)/100,false,1.5/scale);c.restore();}
}
function drawDetail(){drawCrop($('detail-canvas'),1100,650,$('detail-mask').checked,Number($('detail-zoom').value));}
function openRegion(id=selected){if(!current||current.kind==='video'||!regions().some(r=>r.id===id))return;selected=id;renderRegions();draw();$('detail-title').textContent=`Область ${selected} · ${current.filename}`;$('detail-zoom').value='1';$('detail-zoom-value').textContent='1×';drawDetail();if(!$('detail-dialog').open)$('detail-dialog').showModal();}
$('expand-region').addEventListener('click',()=>openRegion());
$('close-detail').addEventListener('click',()=>$('detail-dialog').close());
$('detail-mask').addEventListener('change',drawDetail);$('detail-zoom').addEventListener('input',()=>{$('detail-zoom-value').textContent=$('detail-zoom').value+'×';drawDetail();});
function renderMetadata(){
  const a=current.analysis;
  const rows=[['Источник',current.source==='fixture'?'Учебный пример':'Локальный файл'],['Размер',`${current.width} × ${current.height} px`],['Разметка',a?'Модель'+(current.regions.length?' + ручные отметки':''):current.source==='fixture'?'Задана для демо':'Ручная'],['Модель',a?a.model.checkpoint:'Анализ не выполнялся'],['Автоанализ',a?`${a.inference_ms.toFixed(0)} мс`:'Не выполнялся'],['EXIF / C2PA','Не проверялись']];
  if(a)rows.push(['Порог маски',a.model.threshold],['Устройство',a.model.device],['Весовой файл SHA256',a.model.sha256],['Вход SHA256',a.input_sha256]);
  if(current.kind==='storyboard')rows.push(['Выборка','1 учебный кадр / с'],['Кадров','16'],['Текущий момент',time(second)]);
  $('metadata-panel').innerHTML=`<dl>${rows.map(([a,b])=>`<div class="metadata-row"><dt>${esc(a)}</dt><dd>${esc(b)}</dd></div>`).join('')}</dl><p>Тип изменения и вероятность подделки не определяются. Источник каждой отметки показан отдельно.</p>`;
}
function renderTimeline(){
  $('coverage').textContent=`16 кадров · 1 кадр/с · ${time(current.duration)}`;
  $('timeline-frames').replaceChildren();for(let n=0;n<=15;n++){
    const button=document.createElement('button');button.className=`frame-button ${n===second?'active':''}`;button.setAttribute('aria-label',`Кадр ${time(n)}${n>=5&&n<=10?', учебная область':''}`);button.setAttribute('aria-pressed',n===second);
    button.innerHTML=`<img src="${current.src}" alt=""><span>${time(n)}</span>${n>=5&&n<=10?'<i class="frame-dot"></i>':''}`;
    button.addEventListener('click',()=>{stopPlayback();second=n;render();});$('timeline-frames').append(button);
  }
  const frames=$('timeline-frames'),active=frames.querySelector('.active');
  frames.scrollLeft+=active.getBoundingClientRect().left-frames.getBoundingClientRect().left-frames.clientWidth/2+active.offsetWidth/2;
}
function videoStatus(item){
  const count=item.videoFrames?.length||0,found=(item.videoFrames||[]).some(frame=>(frame.analysis?.area_frac||0)>0||frame.regions.length);
  return {label:item.videoScan?.status==='running'?'Проверка кадров…':item.videoScan?.status==='complete'?'Выборка проверена':count?'Частичная проверка видео':'Видео не проверено',tone:found?'attention':'neutral',detail:`Проверено кадров: ${count}${found?' · есть области для просмотра':''}. Промежутки между кадрами не проверены.`};
}
function renderVideoScan(){
  const owner=videoOwner(),section=$('video-scan');section.hidden=!owner;if(!owner)return;
  const frames=owner.videoFrames||[],scan=owner.videoScan,planned=scan?.planned_times.length||0,running=videoJob?.item===owner;
  $('analyse-video').disabled=current!==owner||!live||!!videoJob||importing||!!loadingItem||caseBusy||!!owner.restored;
  $('analyse-video').textContent=scan?'Проверить видео заново':'Проверить видео';
  $('analyse-video').title=owner.restored?'Для нового анализа загрузите исходный ролик как новый материал':current!==owner?'Сначала вернитесь к ролику':'';
  $('back-video').hidden=current===owner;$('back-video').disabled=!!videoJob||caseBusy;
  $('video-scan-progress').textContent=planned?`${frames.length} / ${planned} кадров · ${videoStatus(owner).label}`:'До 12 кадров · ролик до 60 секунд';
  $('video-scan-bar').hidden=!running;$('video-scan-bar').max=planned||1;$('video-scan-bar').value=frames.length;
  $('video-scan-note').textContent=(scan?.error?scan.error+' ':'')+(owner.restored?'Сохранённые результаты; новый анализ не запускался. ':'')+'Проверены только указанные моменты. Промежутки между кадрами не проверены; пустые маски не подтверждают подлинность ролика.';
  $('video-scan-frames').replaceChildren();
  frames.forEach(frame=>{
    if(!frame.src)frame.src=URL.createObjectURL(frame.file);
    const button=document.createElement('button');button.className=`frame-button ${frame===current?'active':''}`;button.disabled=!!videoJob||caseBusy;
    const found=(frame.analysis?.area_frac||0)>0||frame.regions.length;
    button.setAttribute('aria-label',`Кадр ${frameTime(frame.captured_at)} · ${found?'есть области':'маска пуста'}`);button.setAttribute('aria-pressed',frame===current);
    button.innerHTML=`<img src="${frame.src}" alt=""><span>${frameTime(frame.captured_at)}</span><small>${found?'Есть области':'Маска пуста'}</small><small>${esc(reviewLabels[frame.reviewDraft?.decision]||reviewLabels.pending)}</small>${found?'<i class="frame-dot"></i>':''}`;
    button.addEventListener('click',()=>openItem(frame));$('video-scan-frames').append(button);
  });
}
$('back-video').addEventListener('click',()=>{const owner=videoOwner();if(owner)openItem(owner);});
$('analyse-video').addEventListener('click',async()=>{
  const item=videoOwner();if(!item||current!==item||!live||item.restored||videoJob||importing||loadingItem||caseBusy)return;
  if(item.videoFrames?.length&&!window.confirm('Проверить ролик заново? Текущие результаты кадров и их несохранённые отметки будут заменены.'))return;
  for(const frame of item.videoFrames||[])releaseItem(frame);
  const job={item,controller:new AbortController()};videoJob=job;$('uploaded-video').pause();$('cancel-note').hidden=true;
  renderQueue();renderVideoScan();updateSaveControls();
  try{await analyseVideo(item,{analyseFrame:analyse,signal:job.controller.signal,onUpdate:()=>{if(item.removed)return;caseChanged();renderQueue();renderVideoScan();if(current===item){$('status-title').textContent=videoStatus(item).label;$('status-detail').textContent=videoStatus(item).detail;}}});if(!item.removed)toast(item.videoScan.status==='complete'?`Проверено кадров: ${item.videoFrames.length}. Откройте находки на временной шкале.`:item.videoScan.error||'Проверка отменена. Готовые кадры сохранены.');}
  catch(error){if(error.name!=='AbortError')toast(error.message);}
  finally{if(videoJob===job)videoJob=null;if(!item.removed&&current)render();else renderQueue();updateSaveControls();renderVideoScan();}
});

function caseChanged(){caseRevision++;$('case-save-status').textContent='Есть несохранённые изменения';renderCase();}
function caseDraft(){if(current)current.reviewDraft=activeReview();return {...activeCase,title:$('case-title').value.trim(),note:$('case-note').value.trim(),decision:$('case-decision').value,items:uploaded};}
function fillCase(){for(const key of ['title','note','decision'])$('case-'+key).value=activeCase[key];$('case-save-status').textContent='Все материалы сессии входят в обращение';renderCase();}
function renderCase(){
  if(current)current.reviewDraft=activeReview();
  const summary=caseSummary({items:uploaded}),labels={empty:'Нет материалов',in_progress:'Есть незавершённые проверки',needs_review:'Нужен просмотр оператора',reviewed:'Решения оператора записаны'};
  $('case-summary').textContent=`${labels[summary.status]||summary.status} · материалов ${summary.materials} · проверено ${summary.analysed} · с областями ${summary.findings} · нерешённых ${summary.unresolved}`;
  const busy=caseBusy||projectBusy||importing||!!loadingItem||!!videoJob||explanationBusy;
  for(const id of ['save-case','download-case','case-report'])$(id).disabled=busy||!uploaded.length;
  $('open-case').disabled=busy;
  $('case-materials').replaceChildren();
  uploaded.forEach(item=>{const button=document.createElement('button');button.className=item===current||item.id===current?.parentVideoId?'active':'';button.disabled=busy;
    const review=item===current?activeReview():item.reviewDraft||{decision:'pending'};
    button.innerHTML=`${esc(item.filename)}<small>${esc(item.error?'Ошибка: '+item.error:item.pending||item.queued?'Ожидание':item.cancelled?'Отменено':item.kind==='video'?videoStatus(item).label:item.analysis?statusFor(item,frameRegions(item)).label:'Без анализа')} · ${esc(reviewLabels[review.decision]||reviewLabels.pending)}</small>`;
    button.addEventListener('click',()=>openItem(item));$('case-materials').append(button);
  });
  $('case-page-heading').textContent=activeCase.title.trim()||'Карточка обращения';
  $('case-overview').innerHTML=[['Материалов',summary.materials],['Проверено',summary.analysed],['С областями',summary.findings],['Нерешённых',summary.unresolved]].map(([label,value])=>`<div><strong>${value}</strong><span>${label}</span></div>`).join('');
  $('case-empty').hidden=uploaded.length>0;
  $('case-add-files').disabled=busy;
  $('review-case-caption').textContent=uploaded.length?`Обращение: ${activeCase.title.trim()||'Новое обращение'}`:'Изображение, находки и решение по одному материалу.';
  renderCurrentCase(summary);
  renderMaterialNavigation();
}
for(const key of ['title','note','decision'])$('case-'+key).addEventListener('input',()=>{activeCase[key]=$('case-'+key).value;caseChanged();});
function snapshotItem(item){
  const result={};for(const key of ['id','filename','kind','source','width','height','file','duration','video_time','captured_at','parentVideoId','pending','error','cancelled','queued'])if(item[key]!=null)result[key]=item[key];
  for(const key of ['analysis','regions','explanations','reviewDraft','videoScan'])if(item[key]!=null)result[key]=structuredClone(item[key]);
  if(item.reference)result.reference={file:item.reference.file,width:item.reference.width,height:item.reference.height};
  if(item.videoFrames)result.videoFrames=item.videoFrames.map(snapshotItem);
  if(item===current&&item.kind==='video')result.video_time=$('uploaded-video').currentTime;
  return result;
}
async function saveActiveCase(portable=false){
  if(caseBusy||importing||loadingItem||videoJob||explanationBusy||!uploaded.length)return;
  const draft=caseDraft(),revision=caseRevision;
  caseBusy=true;renderCase();updateSaveControls();
  try{const project=await serializeCase({...draft,items:draft.items.map(snapshotItem)});
    if(portable){const blob=projectBlob(project);download(blob,`${draft.title||'Обращение'}.detective-case.json`);}
    else{await saveCase(project);await refreshCaseHistory();}
    $('case-save-status').textContent=revision===caseRevision?(portable?'Обращение скачано со всеми материалами':'Обращение сохранено в этом браузере'):'Есть несохранённые изменения';
    toast(portable?'Обращение скачано. Включены исходники, кадры, маски и решения.':'Обращение сохранено. Откройте его из истории.');
  }catch(error){toast(`Не удалось сохранить обращение: ${error.message}`);}
  finally{caseBusy=false;renderCase();updateSaveControls();renderVideoScan();}
}
$('save-case').addEventListener('click',()=>saveActiveCase());$('download-case').addEventListener('click',()=>saveActiveCase(true));
$('open-case').addEventListener('click',()=>$('case-input').click());
async function restoreCase(value){
  if(importing||loadingItem||videoJob||caseBusy)throw new Error('Дождитесь завершения операций или отмените ожидание');
  if(uploaded.length&&!window.confirm('Открыть обращение вместо текущей сессии? Несохранённые изменения будут заменены. История останется.'))return;
  ++loadToken;requestController?.abort();explanationController?.abort();stopPlayback();$('uploaded-video').pause();$('uploaded-video').removeAttribute('src');$('uploaded-video').load();
  uploaded.forEach(releaseItem);uploaded.length=0;batchProgress=null;renderEmpty();
  activeCase={id:value.id,title:value.title,note:value.note,decision:value.decision};
  uploaded.push(...value.items);fillCase();renderQueue();
  if(uploaded[0])await openItem(uploaded[0],{navigate:false});
  showPage('case-page');$('case-save-status').textContent='Сохранённое обращение · новый анализ не запускался';
}
$('case-input').addEventListener('change',async e=>{const file=e.target.files[0];if(!file)return;const token=++restoreToken;try{if(file.size>300*1024*1024)throw new Error('Файл обращения превышает 300 МБ');const value=await deserializeCase(JSON.parse(await file.text()));if(token===restoreToken)await restoreCase(value);}catch(error){if(token===restoreToken)toast(`Не удалось открыть обращение: ${error.message}`);}finally{e.target.value='';}});
async function refreshCaseHistory(){
  try{caseSaved=await listCases();renderCaseHistory();}catch(error){$('case-history').textContent=`История обращений недоступна: ${error.message}`;}
}
function renderCaseHistory(){
  $('history-count').textContent=caseSaved.length+saved.length;
  const query=$('case-search').value.trim().toLocaleLowerCase('ru'),status=$('case-filter').value;
  const entries=caseSaved.filter(entry=>(!query||`${entry.title} ${entry.note}`.toLocaleLowerCase('ru').includes(query))&&(!status||entry.status===status));
  $('cases-count').textContent=caseSaved.length?`(${entries.length} / ${caseSaved.length})`:'';
  $('case-history').replaceChildren();if(!entries.length){$('case-history').textContent=caseSaved.length?'По выбранным условиям ничего не найдено.':'Сохранённых обращений пока нет. Создайте новое или откройте файл обращения.';return;}
  entries.forEach(entry=>{const article=document.createElement('article');article.className='case-list-card';article.innerHTML=`<span class="case-list-status ${esc(entry.status)}">${esc(caseStatusLabels[entry.status]||'Сохранено')}</span><h3>${esc(entry.title)}</h3><p>Материалов: ${entry.materials} · проверено: ${entry.analysed} · нерешённых: ${entry.unresolved}</p><p>${esc(entry.note||reviewLabels[entry.decision]||'Без комментария')}</p><time>${esc(new Date(entry.saved_at).toLocaleString('ru-RU'))}</time>`;
    const open=document.createElement('button');open.className='small-button';open.textContent='Открыть обращение';open.addEventListener('click',async()=>{const token=++restoreToken;try{const value=await loadCase(entry.id);if(token!==restoreToken)return;if(!value)throw new Error('Обращение не найдено');await restoreCase(value);}catch(error){toast(error.message);}});
    const remove=document.createElement('button');remove.className='text-button danger';remove.textContent='Удалить';remove.setAttribute('aria-label',`Удалить сохранённое обращение ${entry.title}`);remove.addEventListener('click',async()=>{if(!window.confirm('Удалить сохранённое обращение из истории? Текущая сессия и скачанные файлы останутся.'))return;try{await deleteCase(entry.id);await refreshCaseHistory();}catch(error){toast(error.message);}});
    const actions=document.createElement('div');actions.className='case-list-actions';actions.append(open,remove);article.append(actions);$('case-history').append(article);
  });
}
function renderCurrentCase(summary){
  const hasDraft=uploaded.length||activeCase.title!=='Новое обращение'||activeCase.note||activeCase.decision!=='pending';
  const host=$('current-case');host.replaceChildren();
  if(!hasDraft){host.innerHTML='<div class="current-case-empty"><strong>Начните новую проверку</strong><p>Загрузите фотографию одним действием или создайте обращение для нескольких материалов.</p></div>';return;}
  const card=document.createElement('article');card.className='case-list-card';
  card.innerHTML=`<span class="case-list-status ${esc(summary.status)}">Открыто сейчас · ${esc(caseStatusLabels[summary.status])}</span><h3>${esc(activeCase.title.trim()||'Новое обращение')}</h3><p>Материалов: ${summary.materials} · проверено: ${summary.analysed} · нерешённых: ${summary.unresolved}</p><p>${esc(activeCase.note||'Данные текущей сессии. Сохраните обращение, чтобы открыть его после перезагрузки.')}</p>`;
  const button=document.createElement('button');button.className='small-button';button.textContent='Продолжить обращение';button.addEventListener('click',()=>showPage('case-page'));card.append(button);host.append(card);
}
function renderMaterialNavigation(){
  const owner=videoOwner()||current,index=uploaded.indexOf(owner),busy=importing||!!loadingItem||!!videoJob||caseBusy||projectBusy;
  $('material-position').textContent=index>=0?`Материал ${index+1} из ${uploaded.length}`:current?.source==='fixture'?'Учебный пример':'Материал не выбран';
  $('material-switch').innerHTML='<option value="">Выбрать материал</option>'+uploaded.map(item=>`<option value="${esc(item.id)}">${esc(item.filename)}</option>`).join('');
  $('material-switch').value=index>=0?owner.id:'';$('material-switch').disabled=busy||!uploaded.length;
  $('previous-material').disabled=busy||index<=0;$('next-material').disabled=busy||index<0||index>=uploaded.length-1;
}
$('case-search').addEventListener('input',renderCaseHistory);$('case-filter').addEventListener('change',renderCaseHistory);
$('material-switch').addEventListener('change',e=>{const item=uploaded.find(item=>item.id===e.target.value);if(item)openItem(item);});
for(const [id,direction] of [['previous-material',-1],['next-material',1]])$(id).addEventListener('click',()=>{const index=uploaded.indexOf(videoOwner()||current),item=uploaded[index+direction];if(index>=0&&item)openItem(item);});
function renderQueue(){
  $('queue').hidden=!uploaded.length&&!batchProgress&&!explanationBusy;$('queue-count').textContent=uploaded.length;
  $('cancel-jobs').disabled=!importing&&!loadingItem&&!explanationBusy&&!videoJob;
  $('clear-session').disabled=!uploaded.length||caseBusy||projectBusy;
  $('capture-frame').disabled=importing||!!videoJob||caseBusy;
  $('batch-progress').hidden=!batchProgress;
  if(batchProgress){$('batch-progress').max=batchProgress.total||1;$('batch-progress').value=batchProgress.done;$('session-progress').textContent=`${batchProgress.done} / ${batchProgress.total} завершено${batchProgress.failed?' · ошибок '+batchProgress.failed:''}${batchProgress.cancelled?' · отменено '+batchProgress.cancelled:''} · ${(((batchProgress.finished||Date.now())-batchProgress.started)/1000).toFixed(0)} с`;}
  else $('session-progress').textContent='';
  $('queue-items').replaceChildren();uploaded.forEach(item=>{
    const row=document.createElement('div');row.className='queue-row';
    const button=document.createElement('button');button.className=`queue-item ${item.id===current?.id?'active':''}`;button.title=item.error||'';button.disabled=importing;
    button.disabled=importing||!!videoJob||caseBusy;button.innerHTML=`${icon(item.kind==='video'?'play':'image')}<span>${esc(item.filename)}</span><small>${item.pending?'Проверка…':item.queued?'В очереди':item.cancelled?'Отменено':item.error?'Ошибка':item.kind==='video'?videoStatus(item).label:item.restored?'Сохранено':item.analysis?'Готово':'Открыть'}</small>`;button.addEventListener('click',()=>openItem(item));
    const remove=document.createElement('button');remove.className='queue-remove icon-button';remove.setAttribute('aria-label',`Удалить ${item.filename} из сессии`);remove.title='Удалить из сессии';remove.innerHTML=icon('close');remove.addEventListener('click',()=>removeItem(item));
    remove.disabled=caseBusy||projectBusy;
    row.append(button,remove);$('queue-items').append(row);
  });
  renderCase();
}

function releaseItem(item){item.removed=true;referenceComparison.release(item);for(const frame of item.videoFrames||[])releaseItem(frame);if(item.src?.startsWith('blob:'))URL.revokeObjectURL(item.src);item.element=null;item.maskImage=null;item.colorMask=null;}
function removeItem(item){
  const index=uploaded.indexOf(item);if(index<0)return;
  if(item.queued&&batchProgress){item.queued=false;batchProgress.cancelled++;batchProgress.done++;}
  if(item===loadingItem||loadingItem?.parentVideoId===item.id)requestController?.abort();
  if(item===explanationItem||explanationItem?.parentVideoId===item.id)explanationController?.abort();
  if(item===videoJob?.item)videoJob.controller.abort();
  uploaded.splice(index,1);releaseItem(item);
  if(current===item||current?.parentVideoId===item.id){const video=$('uploaded-video');if(item.kind==='video'&&video.src===item.src){video.pause();video.removeAttribute('src');video.load();}renderEmpty();}
  caseChanged();
  renderQueue();
}
function cancelJobs(){
  batchGeneration++;restoreToken++;requestController?.abort();explanationController?.abort();videoJob?.controller.abort();
  for(const item of uploaded)if(item.queued){item.queued=false;item.cancelled=true;if(batchProgress){batchProgress.cancelled++;batchProgress.done++;}}
  $('cancel-note').hidden=false;renderQueue();
}
$('cancel-jobs').addEventListener('click',cancelJobs);
function clearSession(){
  if(caseBusy||projectBusy){toast('Дождитесь завершения сохранения');return false;}
  const hasDraft=uploaded.length||activeCase.title!=='Новое обращение'||activeCase.note||activeCase.decision!=='pending';
  if(hasDraft&&!window.confirm('Начать пустую сессию? Несохранённые изменения будут удалены. Сохранённая история останется.'))return false;
  cancelJobs();++loadToken;loadingItem=null;stopPlayback();$('uploaded-video').pause();$('uploaded-video').removeAttribute('src');$('uploaded-video').load();
  uploaded.forEach(releaseItem);uploaded.length=0;batchProgress=null;$('loading').hidden=true;$('cancel-note').hidden=true;renderEmpty();renderQueue();
  activeCase={id:crypto.randomUUID(),title:'Новое обращение',note:'',decision:'pending'};fillCase();
  return true;
}
$('clear-session').addEventListener('click',clearSession);
$('new-case').addEventListener('click',()=>{if(clearSession())showPage('case-page');});
$('quick-check').addEventListener('click',()=>{if(clearSession()){showPage('workspace');$('file-input').click();}});
$('case-add-files').addEventListener('click',()=>$('file-input').click());

document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>{cancelGesture();view=button.dataset.view;document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('selected',b===button));$('compare-control').hidden=view!=='compare';viewerNeedsLayout=true;draw();}));
document.querySelectorAll('[data-tab]').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('[data-tab]').forEach(b=>b.classList.toggle('active',b===button));for(const panel of ['findings','explanation','metadata'])$(panel+'-panel').hidden=button.dataset.tab!==panel;}));
$('opacity').addEventListener('input',()=>{$('opacity-value').textContent=$('opacity').value+'%';draw();});$('compare-slider').addEventListener('input',draw);
$('zoom-in').addEventListener('click',()=>{cancelGesture();zoom=Math.min(3,zoom+.25);draw();});$('zoom-out').addEventListener('click',()=>{cancelGesture();zoom=Math.max(1,zoom-.25);draw();});$('zoom-reset').addEventListener('click',()=>{cancelGesture();zoom=1;viewerNeedsLayout=true;resetScroll=true;draw();});
$('mark-button').addEventListener('click',()=>{cancelGesture();marking=!marking;render();if(marking)toast('Протяните прямоугольник на изображении. Отметка будет ручной.');});
function coordinates(e){return imagePoint(e.clientX,e.clientY,canvas.getBoundingClientRect(),current.width,current.height);}
function hitRegion(point){
  if(view==='original'||view==='compare'&&point[0]<current.width*Number($('compare-slider').value)/100)return null;
  const list=regions(),polygon=[...list].reverse().find(r=>r.polygon&&pointInPolygon(point,r.polygon));if(polygon)return polygon;
  if(!current.analysis||!current.maskImage)return null;
  const x=Math.min(current.width-1,Math.floor(point[0])),y=Math.min(current.height-1,Math.floor(point[1]));
  if(!colorMask(current).getContext('2d').getImageData(x,y,1,1).data[3])return null;
  // shortcut: overlapping component boxes use the smallest match; add component labels only if exact overlap selection is needed.
  return list.filter(r=>!r.polygon&&point[0]>=r.x&&point[0]<r.x+r.width&&point[1]>=r.y&&point[1]<r.y+r.height).sort((a,b)=>a.width*a.height-b.width*b.height)[0]||null;
}
function cancelGesture(){const old=gesture;gesture=null;drawStart=null;draft=null;$('compare-dragger').classList.remove('dragging');if(old?.target.hasPointerCapture(old.id))old.target.releasePointerCapture(old.id);}
function moveDivider(e){$('compare-slider').value=dividerPercent(e.clientX,canvas.getBoundingClientRect());draw();}
function pointerDown(e){
  if(!current||current.kind==='video'||loadingItem||gesture||e.button!==0||e.isPrimary===false)return;
  const point=coordinates(e),mode=marking?'mark':view==='compare'&&(e.currentTarget===$('compare-dragger')||nearDivider(e.clientX,canvas.getBoundingClientRect(),Number($('compare-slider').value),20))?'compare':'click';
  gesture={id:e.pointerId,target:e.currentTarget,item:current,mode,view,startX:e.clientX,startY:e.clientY,region:mode==='click'?hitRegion(point)?.id:null,moved:false};
  e.currentTarget.setPointerCapture(e.pointerId);
  if(mode==='mark'){drawStart=point;e.preventDefault();}
  if(mode==='compare'){$('compare-dragger').classList.add('dragging');e.preventDefault();moveDivider(e);}
}
function pointerMove(e){
  if(!gesture||gesture.id!==e.pointerId)return;
  if(gesture.item!==current||gesture.view!==view){cancelGesture();return;}
  if(Math.hypot(e.clientX-gesture.startX,e.clientY-gesture.startY)>6)gesture.moved=true;
  if(gesture.mode==='compare'){e.preventDefault();moveDivider(e);}
  else if(gesture.mode==='mark'){const point=coordinates(e);draft={x:Math.min(point[0],drawStart[0]),y:Math.min(point[1],drawStart[1]),w:Math.abs(point[0]-drawStart[0]),h:Math.abs(point[1]-drawStart[1])};e.preventDefault();draw();}
}
function pointerUp(e){
  if(!gesture||gesture.id!==e.pointerId)return;
  const old=gesture,rectangle=draft,same=old.item===current&&old.view===view,moved=old.moved||Math.hypot(e.clientX-old.startX,e.clientY-old.startY)>6;
  const hit=same&&old.mode==='click'&&!moved?hitRegion(coordinates(e)):null;
  cancelGesture();if(!same)return;
  if(old.mode==='compare'){moveDivider(e);return;}
  if(old.mode==='mark'){
    if(rectangle&&rectangle.w>8&&rectangle.h>8){const {x,y,w,h}=rectangle,id=Math.max(0,...regions().map(r=>r.id))+1;current.regions.push({id,title:`Ручная область ${id}`,hint:'Эта область отмечена пользователем для дальнейшей проверки. Причина изменения и наличие манипуляции автоматически не определялись.',manual:true,polygon:[[x,y],[x+w,y],[x+w,y+h],[x,y+h]]});selected=id;$('save-status').textContent='Есть несохранённые изменения';toast('Ручная отметка добавлена');}
    marking=false;caseChanged();render();return;
  }
  if(hit&&hit.id===old.region)openRegion(hit.id);
}
for(const target of [canvas,$('compare-dragger')]){
  target.addEventListener('pointerdown',pointerDown);target.addEventListener('pointermove',pointerMove);target.addEventListener('pointerup',pointerUp);
  for(const event of ['pointercancel','lostpointercapture'])target.addEventListener(event,e=>{if(gesture?.id===e.pointerId){cancelGesture();draw();}});
}
window.addEventListener('resize',()=>{cancelGesture();viewerNeedsLayout=true;draw();});
const viewerObserver=new ResizeObserver(()=>{const {width,height}=$('canvas-scroll').getBoundingClientRect();if(width!==observedViewerWidth||height!==observedViewerHeight){observedViewerWidth=width;observedViewerHeight=height;if(width&&height){cancelGesture();viewerNeedsLayout=true;draw();}}});viewerObserver.observe($('canvas-scroll'));
$('remove-region').addEventListener('click',()=>{current.regions=current.regions.filter(r=>r.id!==selected);$('save-status').textContent='Есть несохранённые изменения';caseChanged();render();});

async function addFiles(files){
  if(importing||videoJob||caseBusy){toast('Дождитесь обработки материалов или отмените ожидание');return;}
  const available=20-uploaded.length;if(available<=0){toast('В одной сессии можно открыть до 20 файлов');return;}
  const items=Array.from(files).slice(0,available);if(files.length>available)toast('Часть файлов пропущена: лимит 20 материалов');
  const added=[];
  for(const file of items){
    const video=['video/mp4','video/webm'].includes(file.type),image=['image/jpeg','image/png','image/webp'].includes(file.type);
    if(!video&&!image){toast(`Не поддерживается: ${file.name}. Выберите JPEG, PNG, WebP, MP4 или WebM.`);continue;}
    if(file.size>(video?100:20)*1024*1024){toast(`Файл ${file.name} слишком большой: лимит ${video?100:20} МБ`);continue;}
    const item={id:crypto.randomUUID(),filename:file.name,file,kind:video?'video':'image',source:'upload',src:URL.createObjectURL(file),regions:[],width:0,height:0,queued:true};
    uploaded.push(item);added.push(item);
  }
  if(!added.length)return;
  if(activePage==='history')showPage('workspace');
  caseChanged();
  const generation=++batchGeneration,progress={total:added.length,done:0,failed:0,cancelled:0,started:Date.now()};batchProgress=progress;$('cancel-note').hidden=true;
  importing=true;$('upload-button').disabled=true;$('select-files').disabled=true;
  renderQueue();
  try{for(const item of added){if(generation!==batchGeneration)break;if(item.removed)continue;const ok=await openItem(item,{batch:true,navigate:false});progress.done++;if(!ok){if(item.cancelled||item.removed)progress.cancelled++;else progress.failed++;}renderQueue();}}
  finally{progress.finished=Date.now();importing=false;$('upload-button').disabled=false;$('select-files').disabled=false;$('file-input').value='';renderExplanation();renderQueue();updateSaveControls();renderVideoScan();}
}
['upload-button','select-files','empty-upload'].forEach(id=>$(id).addEventListener('click',()=>$('file-input').click()));
$('file-input').addEventListener('change',e=>addFiles(e.target.files));
const drop=$('drop-zone');drop.addEventListener('dragover',e=>{e.preventDefault();drop.classList.add('over');});drop.addEventListener('dragleave',()=>drop.classList.remove('over'));drop.addEventListener('drop',e=>{e.preventDefault();drop.classList.remove('over');addFiles(e.dataTransfer.files);});
window.addEventListener('dragover',e=>e.preventDefault());window.addEventListener('drop',e=>e.preventDefault());
window.addEventListener('beforeunload',()=>{clearInterval(healthTimer);requestController?.abort();explanationController?.abort();videoJob?.controller.abort();uploaded.forEach(releaseItem);});
$('play-frames').addEventListener('click',()=>{if(playTimer){stopPlayback();return;}$('play-frames').textContent='Пауза';playTimer=setInterval(()=>{second=(second+1)%16;render();},1000);});
$('capture-frame').addEventListener('click',async()=>{
  if(current?.kind!=='video'||caseBusy||projectBusy||videoJob||importing||loadingItem)return;
  if(uploaded.length>=20){toast('Лимит сессии: 20 материалов');return;}
  const owner=current,token=loadToken,caseId=activeCase.id,video=$('uploaded-video');if(video.readyState<2){toast('Дождитесь загрузки кадра видео');return;}
  video.pause();$('capture-frame').disabled=true;const c=document.createElement('canvas');c.width=owner.width;c.height=owner.height;
  try{c.getContext('2d').drawImage(video,0,0);const timestamp=video.currentTime,filename=`${owner.filename} · ${frameTime(timestamp)}.png`,blob=await new Promise(resolve=>c.toBlob(resolve,'image/png'));
    if(current!==owner||owner.removed||token!==loadToken||caseId!==activeCase.id||caseBusy||projectBusy||videoJob||importing||loadingItem||uploaded.length>=20)return;
    if(!blob)throw new Error('Не удалось извлечь кадр');
    const file=new File([blob],filename,{type:'image/png'}),item={id:crypto.randomUUID(),filename,file,kind:'image',source:'upload',src:URL.createObjectURL(file),regions:[],width:c.width,height:c.height,captured_at:timestamp};uploaded.push(item);caseChanged();await openItem(item);
  }catch(error){toast(error.message);}finally{c.width=c.height=0;renderQueue();}
});

const pageRoutes={history:'cases','case-page':'case',workspace:'review',integrations:'integrations',guide:'guide'};
const pageTitles={history:'Обращения','case-page':'Карточка обращения',workspace:'Проверка материала',integrations:'API и MCP',guide:'Как читать результат'};
function routePage(){return Object.keys(pageRoutes).find(page=>pageRoutes[page]===location.hash.slice(1))||'history';}
function showPage(page,{updateHash=true}={}){
  if(!(page in pageRoutes))page='history';
  const changed=activePage!==page;activePage=page;
  if(page!=='workspace'){stopPlayback();$('uploaded-video').pause();cancelGesture();}
  Object.keys(pageRoutes).forEach(id=>$(id).hidden=id!==page);
  const navPage=page==='case-page'?'history':page;
  document.querySelectorAll('[data-nav]').forEach(button=>{const active=button.dataset.nav===navPage;button.classList.toggle('active',active);if(active)button.setAttribute('aria-current','page');else button.removeAttribute('aria-current');});
  $('page-label').textContent=pageTitles[page];document.title=pageTitles[page]+' — Цифровой детектив';
  if(page==='history'){renderHistory();renderCaseHistory();}
  if(page==='case-page')renderCase();
  if(page==='workspace'){renderMaterialNavigation();viewerNeedsLayout=true;requestAnimationFrame(draw);}
  if(updateHash&&location.hash!=='#'+pageRoutes[page])location.hash=pageRoutes[page];
  if(changed){window.scrollTo(0,0);$('main').focus({preventScroll:true});}
}
window.addEventListener('hashchange',()=>showPage(routePage(),{updateHash:false}));
document.querySelector('.skip-link').addEventListener('click',e=>{e.preventDefault();$('main').focus();$('main').scrollIntoView({block:'start'});});
document.querySelectorAll('[data-nav]').forEach(b=>b.addEventListener('click',()=>showPage(b.dataset.nav)));
function persistLegacy(){localStorage.setItem('detective-demo-history',JSON.stringify(legacySaved));}
async function refreshHistory(){
  try{const full=await listReviews();saved=[...full.map(entry=>({...entry,full:true})),...legacySaved.filter(entry=>!full.some(item=>item.id===entry.id))].sort((a,b)=>new Date(b.saved_at)-new Date(a.saved_at));}
  catch(error){saved=legacySaved;$('history-storage-note').textContent=`Полная история недоступна: ${error.message}. Скачайте файл проверки для сохранения.`;}
  renderHistory();
}
function updateSaveControls(){const busy=projectBusy||caseBusy||!!videoJob||!!loadingItem||!!current?.explanationPending;$('save-review').disabled=!current||busy;$('save-project').disabled=!current?.file||busy;$('open-project').disabled=busy||importing;$('mask-download').disabled=!current||current.kind==='video'||!!loadingItem;renderCase();}
const reviewState=(item,review)=>JSON.stringify({review,regions:item.regions,explanations:item.explanations||{},reference:item.reference?{name:item.reference.file.name,size:item.reference.file.size,modified:item.reference.file.lastModified}:null,frames:(item.videoFrames||[]).map(frame=>({id:frame.id,regions:frame.regions,review:frame.reviewDraft,reference:frame.reference?.file.name})),scan:item.videoScan});
async function currentProject(item=current,review=activeReview()){
  if(!item?.file||item.pending||item.error||item.explanationPending)throw new Error('Дождитесь завершения проверки и пояснения перед сохранением');
  return serializeProject(snapshotItem(item),review);
}
$('save-review').addEventListener('click',async()=>{
  if(!current||projectBusy||loadingItem)return;const item=current,review=activeReview(),state=reviewState(item,review);item.reviewDraft=review;projectBusy=true;updateSaveControls();
  try{
    if(item.file){await saveReview(await currentProject(item,review));legacySaved=legacySaved.filter(entry=>entry.id!==item.id);}
    else{const entry={id:item.id,filename:item.filename,fixture_id:item.source==='fixture'?item.id:null,source:item.source,saved_at:new Date().toISOString(),review,region_count:regions().length};legacySaved=[entry,...legacySaved.filter(e=>e.id!==entry.id)].slice(0,30);persistLegacy();}
    await refreshHistory();if(current===item)$('save-status').textContent=reviewState(item,activeReview())!==state?'Есть несохранённые изменения':item.file?'Проверка полностью сохранена в этом браузере':'Решение для учебного примера сохранено';toast(item.file?'Файл, маска, отметки, пояснения и решение сохранены':'Решение и комментарий сохранены');
  }catch(error){toast(`Не удалось сохранить: ${error.message}. Можно скачать файл проверки.`);}
  finally{projectBusy=false;updateSaveControls();}
});
$('save-project').addEventListener('click',async()=>{if(projectBusy||loadingItem)return;const item=current,review=activeReview();projectBusy=true;updateSaveControls();try{const project=await currentProject(item,review);download(projectBlob(project),`${item.filename}.detective.json`);toast('Файл проверки скачан. Он содержит исходный материал и результат.');}catch(error){toast(error.message);}finally{projectBusy=false;updateSaveControls();}});
async function restoreItem(item){
  if(importing||loadingItem||videoJob||caseBusy||projectBusy)throw new Error('Дождитесь обработки файлов или отмените ожидание');
  const existing=uploaded.find(entry=>entry.id===item.id);
  if(existing){if(!window.confirm('Открыть сохранённую версию? Изменения этого материала в текущей сессии будут заменены.'))return;removeItem(existing);}
  if(uploaded.length>=20)throw new Error('Лимит 20 материалов. Удалите материал из сессии.');
  item.src=URL.createObjectURL(item.file);item.restored=true;uploaded.push(item);await openItem(item);
}
$('open-project').addEventListener('click',()=>$('project-input').click());
$('project-input').addEventListener('change',async e=>{const file=e.target.files[0];if(!file)return;const token=++restoreToken;try{if(file.size>300*1024*1024)throw new Error('Файл проверки превышает 300 МБ');const item=await deserializeProject(JSON.parse(await file.text()));if(token===restoreToken)await restoreItem(item);}catch(error){if(token===restoreToken)toast(`Не удалось открыть проверку: ${error.message}`);}finally{e.target.value='';}});
['review-select','review-note'].forEach(id=>$(id).addEventListener('input',()=>{$('save-status').textContent='Есть несохранённые изменения';caseChanged();renderVideoScan();}));
function renderHistory(){
  $('history-count').textContent=caseSaved.length+saved.length;$('history-list').replaceChildren();
  if(!saved.length){$('history-list').innerHTML=`<div class="history-empty">${icon('history')}<h2>Первая проверка — впереди</h2><p>Сохраните решение и комментарий на экране материала.</p></div>`;return;}
  saved.forEach(entry=>{const article=document.createElement('article');article.className='history-entry';article.innerHTML=`<div class="file-icon">${icon('file')}</div><div><h3>${esc(entry.filename)}</h3><p>${esc(reviewLabels[entry.review?.decision]||reviewLabels.pending)} · ${entry.region_count} ${regionWord(entry.region_count)}</p><p>${esc(entry.review?.note||'Без комментария')}</p></div><time>${esc(new Date(entry.saved_at).toLocaleString('ru-RU'))}</time>`;
    const item=fixtures.find(f=>f.id===entry.fixture_id)||uploaded.find(f=>f.id===entry.id);
    const kind=document.createElement('small');kind.className='history-kind';kind.textContent=entry.full?'Полная проверка':'Только решение · прежний формат / учебный пример';article.children[1].append(kind);
    if(entry.full||item){const button=document.createElement('button');button.className='small-button';button.textContent='Открыть';button.addEventListener('click',async()=>{const token=++restoreToken;try{if(entry.full){const restored=await loadReview(entry.id);if(token!==restoreToken)return;if(!restored)throw new Error('Проверка не найдена');await restoreItem(restored);}else await openItem({...item,source:entry.source,regions:structuredClone(item.regions||[])});}catch(error){if(token===restoreToken)toast(error.message);}});article.append(button);}
    const remove=document.createElement('button');remove.className='text-button danger';remove.textContent='Удалить';remove.setAttribute('aria-label',`Удалить сохранённую проверку ${entry.filename}`);remove.addEventListener('click',async()=>{if(!window.confirm('Удалить эту проверку из истории браузера? Текущая сессия и скачанные файлы останутся.'))return;try{if(entry.full)await deleteReview(entry.id);legacySaved=legacySaved.filter(item=>item.id!==entry.id);persistLegacy();await refreshHistory();}catch(error){toast(error.message);}});article.append(remove);$('history-list').append(article);
  });
}
$('clear-history').addEventListener('click',async()=>{if(!saved.length||!window.confirm('Удалить все сохранённые проверки и решения из этого браузера? Текущая сессия и скачанные файлы останутся.'))return;try{await clearReviews();legacySaved=[];persistLegacy();await refreshHistory();}catch(error){toast(`Не удалось очистить историю: ${error.message}`);}});
$('presentation').addEventListener('click',()=>{const active=document.body.classList.toggle('presentation-mode');$('presentation').setAttribute('aria-label',active?'Выключить режим показа':'Включить режим показа');$('presentation').classList.toggle('active',active);cancelGesture();viewerNeedsLayout=true;requestAnimationFrame(draw);});
$('tour-button').addEventListener('click',()=>$('tour-dialog').showModal());$('close-tour').addEventListener('click',()=>$('tour-dialog').close());$('start-tour').addEventListener('click',()=>{$('tour-dialog').close();document.body.classList.add('demo-examples');openItem({...fixtures[0],source:'fixture',regions:structuredClone(fixtures[0].regions)});});
document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName))return;if(e.key==='Escape'&&document.body.classList.contains('presentation-mode'))$('presentation').click();});

function resultCanvas(maskOnly=false){
  const c=document.createElement('canvas');c.width=current.width;c.height=current.height;const context=c.getContext('2d');
  if(maskOnly&&current.analysis){context.drawImage(current.maskImage,0,0);}
  else if(maskOnly){context.fillStyle='#000';context.fillRect(0,0,c.width,c.height);context.fillStyle='#fff';regions().forEach(r=>{path(context,r.polygon);context.fill();});const image=context.getImageData(0,0,c.width,c.height);for(let i=0;i<image.data.length;i+=4){const value=image.data[i]>=128?255:0;image.data[i]=image.data[i+1]=image.data[i+2]=value;}context.putImageData(image,0,0);}
  else{drawBase(context);drawMask(context,regions(),Number($('opacity').value)/100);}
  return c;
}
function download(blob,name){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function projectBlob(project){const blob=new Blob([JSON.stringify(project)],{type:'application/json'});if(blob.size>300*1024*1024)throw new Error('Архив превышает 300 МБ. Уменьшите число или размер материалов.');return blob;}
function downloadCanvas(c,name){c.toBlob(blob=>{if(blob)download(blob,name);else toast('Не удалось подготовить изображение');},'image/png');}
$('mask-download').addEventListener('click',()=>{if(current)downloadCanvas(resultCanvas(true),'detective-mask.png');});
function reportMarkup(){
  const data=reportData(current,regions(),activeReview(),current.kind==='video'?$('uploaded-video').currentTime:second);
  const image=resultCanvas().toDataURL('image/png');
  const vlm=(data.explanations.length?`<h3>Пояснения VLM · гипотезы</h3>${data.explanations.map(r=>`<section><h4>Область ${r.region_id}</h4><dl><dt>Объект</dt><dd>${esc(r.description.object)}</dd><dt>Наблюдение</dt><dd>${esc(r.description.observation)}</dd><dt>Подсказка / возможная правка</dt><dd>${esc(r.description.possible_change)}</dd><dt>Тип операции</dt><dd>${esc(operationLabels[r.description.operation])}</dd><dt>Основания</dt><dd>${esc(r.description.evidence.join(' · ')||'Недостаточно данных')}</dd><dt>Ограничения</dt><dd>${esc(r.description.limitations)}</dd></dl><p>${esc(r.limitation)}</p><small>OpenRouter · ${esc(r.model)} · ${esc(r.generated_at)} · ${esc(r.prompt_version)}</small></section>`).join('')}`:'')+reportExtras(data);
  return `<article class="report-content"><div class="report-label">ЦИФРОВОЙ ДЕТЕКТИВ / КАРТОЧКА ПРОВЕРКИ</div><h2>${esc(data.filename)}</h2><div class="report-disclaimer">${esc(data.limitation)}</div><dl><dt>Дата</dt><dd>${new Date(data.generated_at).toLocaleString('ru-RU')}</dd><dt>Источник</dt><dd>${current.source==='fixture'?'Учебный пример':'Локальный файл'}</dd><dt>Размер</dt><dd>${data.width} × ${data.height} px</dd><dt>Модель</dt><dd>${data.analysis_performed?esc(data.model.checkpoint)+' · '+esc(data.model.precision):'Анализ не выполнялся'}</dd>${data.analysis_performed?`<dt>SHA256 весов</dt><dd>${esc(data.model.sha256)}</dd><dt>SHA256 входа</dt><dd>${esc(data.input_sha256)}</dd><dt>Порог маски</dt><dd>${data.model.threshold}</dd><dt>Время модели</dt><dd>${data.inference_ms} мс</dd>`:''}${data.timestamp_seconds!==null?`<dt>Кадр</dt><dd>${time(data.timestamp_seconds)}</dd>`:''}<dt>Результат просмотра</dt><dd>${esc(data.status)}</dd></dl><img src="${image}" alt="Материал с маской или ручной разметкой"><table><thead><tr><th>Область</th><th>Доля кадра</th><th>Наблюдение</th></tr></thead><tbody>${data.regions.map(r=>`<tr><td>${r.id}. ${esc(r.title)}</td><td>${percent(r.area_frac)}</td><td>${esc(r.hint)}</td></tr>`).join('')||'<tr><td colspan="3">Области не выделены. Это не подтверждение подлинности.</td></tr>'}</tbody></table>${vlm}<h3>Решение оператора</h3><p>${esc(reviewLabels[data.review.decision])}</p><p>${esc(data.review.note||'Комментарий не добавлен.')}</p><small>LOCAL · v0.2 · вывод оператора отделён от автоматического анализа</small></article>`;
}
function reportExtras(data){return (data.reference?`<h3>Сравнение с исходником</h3><p>${esc(data.reference.filename)} · ${data.reference.width} × ${data.reference.height} px. Визуальное сравнение без автоматического совмещения. Исходник не проверялся моделью.</p>`:'')+(data.video?`<h3>Проверка кадров видео</h3><p>${esc(scanLabels[data.video.scan?.status]||'Не запускалась')} · проверено кадров: ${data.video.frames.length}. ${esc(data.video.coverage)} ${esc(data.video.scan?.error||'')}</p><table><thead><tr><th>Момент</th><th>Маска</th><th>Решение оператора</th></tr></thead><tbody>${data.video.frames.map(frame=>`<tr><td>${frameTime(frame.timestamp_seconds)}</td><td>${frame.model_region_count} обл. · ${percent(frame.model_mask_area_frac||0)}</td><td>${esc(reviewLabels[frame.review.decision])}<br>${esc(frame.review.note)}</td></tr>`).join('')}</tbody></table>`:'');}
function showReport(data,markup){reportSnapshot={data,markup};$('report-dialog').querySelector('.dialog-toolbar strong').textContent=data.case?'Общий отчёт по обращению':'Карточка проверки';$('report-content').innerHTML=markup;$('download-overlay').hidden=!!data.case;$('report-dialog').showModal();}
function caseReportData(){const draft=caseDraft();return {format:'detective-case-report',format_version:1,generated_at:new Date().toISOString(),case:{id:draft.id,title:draft.title,note:draft.note,decision:draft.decision},summary:caseSummary(draft),materials:uploaded.map(item=>({...reportData(item,frameRegions(item),item.reviewDraft||{decision:'pending',note:''},item.video_time||0),error:item.error||null,cancelled:!!item.cancelled})),limitation:'Находки модели и решения оператора приведены отдельно. Пустая маска не удостоверяет подлинность. Для видео проверены только перечисленные моменты.'};}
function caseReportMarkup(data){return `<article class="report-content"><div class="report-label">ЦИФРОВОЙ ДЕТЕКТИВ / ОБРАЩЕНИЕ</div><h2>${esc(data.case.title)}</h2><div class="report-disclaimer">${esc(data.limitation)}</div><p>${esc(new Date(data.generated_at).toLocaleString('ru-RU'))} · материалов ${data.summary.materials} · проверено ${data.summary.analysed} · нерешённых ${data.summary.unresolved}</p><h3>Общее решение оператора</h3><p>${esc(reviewLabels[data.case.decision])}</p><p>${esc(data.case.note||'Комментарий не добавлен.')}</p>${data.materials.map((item,i)=>`<section><h3>${i+1}. ${esc(item.filename)}</h3><p>${esc(item.status)}${item.error?' · Ошибка: '+esc(item.error):''}${item.cancelled?' · Ожидание отменено':''}</p><p>${esc(reviewLabels[item.review.decision])} · ${esc(item.review.note||'Без комментария')}</p>${item.analysis_performed?`<dl><dt>Области модели</dt><dd>${item.model_region_count} · ${percent(item.model_mask_area_frac)}</dd><dt>Порог</dt><dd>${item.model.threshold}</dd><dt>SHA256 весов</dt><dd>${esc(item.model.sha256)}</dd><dt>SHA256 входа</dt><dd>${esc(item.input_sha256)}</dd></dl>`:''}${item.regions.length?`<table class="case-report-table"><thead><tr><th>Область</th><th>Источник</th><th>Координаты</th></tr></thead><tbody>${item.regions.map(r=>`<tr><td>${r.id}. ${esc(r.title)}</td><td>${r.manual?'Ручная':'Модель'}</td><td>${[r.x,r.y,r.width,r.height].map(Math.round).join(', ')}</td></tr>`).join('')}</tbody></table>`:''}${reportExtras(item)}</section>`).join('')}<small>Исходники, маски и полные результаты включаются в файл «Скачать обращение». JSON этого отчёта содержит подробные результаты, включая каждый кадр.</small></article>`;}
$('case-report').addEventListener('click',()=>{if(!uploaded.length)return;const data=caseReportData();showReport(data,caseReportMarkup(data));});
$('report-button').addEventListener('click',()=>{if(!current)return;if(current.kind==='video'&&$('uploaded-video').readyState<2){toast('Дождитесь загрузки кадра видео');return;}const data=reportData(current,regions(),activeReview(),current.kind==='video'?$('uploaded-video').currentTime:second);showReport(data,reportMarkup());});
$('close-report').addEventListener('click',()=>$('report-dialog').close());
$('download-json').addEventListener('click',()=>{if(reportSnapshot)download(new Blob([JSON.stringify(reportSnapshot.data,null,2)],{type:'application/json'}),reportSnapshot.data.case?'detective-case-report.json':'detective-report.json');});
$('download-overlay').addEventListener('click',()=>downloadCanvas(resultCanvas(),'detective-overlay.png'));
$('download-html').addEventListener('click',()=>{if(!reportSnapshot)return;const html=`<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Карточка проверки — Цифровой детектив</title><style>body{font:14px/1.6 Arial,sans-serif;color:#284333;max-width:900px;margin:40px auto;padding:0 24px}img{width:100%}table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #dce4d7;text-align:left;padding:12px}dl{display:grid;grid-template-columns:150px 1fr;gap:8px}dd{margin:0}p{white-space:pre-wrap;overflow-wrap:anywhere}.report-disclaimer{background:#faf0da;padding:15px;border-radius:8px}small{color:#829174}@media print{body{margin:0}img{max-height:350px;object-fit:contain}}</style>${reportSnapshot.markup}</html>`;download(new Blob([html],{type:'text/html;charset=utf-8'}),reportSnapshot.data.case?'detective-case-report.html':'detective-report.html');});
$('print-report').addEventListener('click',()=>window.print());
function renderEmpty(){
  cancelGesture();current=null;$('canvas-scroll').hidden=true;$('uploaded-video').hidden=true;$('compare-dragger').hidden=true;$('image-stage').classList.remove('has-image');
  $('stage-label').parentElement.hidden=true;$('canvas-hint').hidden=true;$('upload-empty').hidden=false;
  $('filename').textContent='Новый материал';$('filemeta').textContent='JPEG, PNG, WebP · до 20 МБ и 15 млн пикселей';$('source-tag').textContent='Ваш файл';
  $('status-title').textContent='Готов к проверке';$('status-detail').textContent='Добавьте изображение. Модель выделит области, которые можно рассмотреть и сохранить.';
  $('status-icon').className='status-icon neutral';$('regions-count').textContent='—';$('area-value').textContent='—';$('findings-count').textContent='0';
  $('regions-word').textContent='областей';$('area-value').title='';$('area-value').nextElementSibling.textContent='доля изображения';$('result-number').textContent='LOCAL';
  $('viewer-note').textContent='Добавьте материал для проверки.';$('dimensions').textContent='';
  $('review-select').value='pending';$('review-note').value='';$('save-status').textContent='Нет открытой проверки';
  $('timeline').hidden=true;$('video-tools').hidden=true;$('region-list-note').hidden=true;
  $('region-list').replaceChildren();$('region-detail').hidden=true;$('no-regions').hidden=false;$('empty-detail').textContent='После проверки здесь появятся найденные области.';
  $('no-regions').querySelector('strong').textContent='Нет выделенных областей';
  $('metadata-panel').replaceChildren();$('report-button').disabled=true;$('mask-download').disabled=true;$('mark-button').disabled=true;
  $('save-project').disabled=true;
  ['save-review','review-select','review-note'].forEach(id=>$(id).disabled=true);
  renderExplanation();renderVideoScan();referenceComparison.render();renderCase();
}

async function refreshHealth(){
  if(healthBusy)return;healthBusy=true;
  const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),3000);let label;
  try{
    const response=await fetch('/health',{signal:controller.signal,cache:'no-store'});
    if(!response.ok)throw new Error(response.status===404?'demo':'offline');
    const health=await response.json();knownModel=health.mode==='inference'||knownModel;live=health.mode==='inference'&&health.status==='ok';device=typeof health.device==='string'?health.device:'';
    explanationService=health.explanations||{available:false,model:explanationService.model};
    renderIntegrationStatus(health);
    label=live?`Модель готова${device?' · '+device:''}`:health.mode==='inference'?'Модель загружается':'Демо · без модели';
  }catch(error){live=false;explanationService={available:false,model:explanationService.model};label=knownModel?'Сервис недоступен':error.message==='demo'?'Демо · без модели':'Нет связи с сервисом';renderIntegrationStatus();}
  finally{clearTimeout(timeout);healthBusy=false;}
  $('mode-badge').innerHTML='<span></span>'+esc(label);
  $('mode-badge').classList.toggle('live',live);document.body.classList.toggle('live-mode',live);
  $('storage-note').textContent=explanationService.available?'Маска локально · пояснения через OpenRouter':live?'Обработка на вашем компьютере':'Файлы остаются в браузере';
  $('upload-caption').textContent=live?'Изображения и кадры коротких видео → локальная модель':'Изображения и видео · без автоматического анализа';
  $('bottom-description').textContent=live?'Реальная локальная модель. Ручные отметки сохраняются отдельно. Учебные примеры содержат заранее заданную разметку.':'Прототип интерфейса. В примерах используется учебная разметка; для ваших файлов доступны просмотр и ручные отметки.';
  $('guide-description').textContent=live?'Ваши изображения проверяет локальная модель SegFormer. Учебные примеры иллюстрируют интерфейс и не запускают модель. Результат проверки и ручные отметки имеют разные источники.':'Нейросеть не подключена. Учебные примеры нарисованы для демонстрации; их маски заданы заранее.';
  $('capture-frame').innerHTML=icon('image')+(live?'Проверить выбранный кадр':'Открыть кадр для отметок');
  renderExplanation();renderVideoScan();if(importing)renderQueue();
}
function renderIntegrationStatus(health){
  const model=health?.model;
  $('integration-status').textContent=health?.status==='ok'&&health.mode==='inference'?'Сервис готов':health?.mode==='inference'?'Загрузка модели':'Нет подключения к модели';
  $('integration-device').textContent=health?.device||'—';
  $('integration-model').textContent=model?.checkpoint||'—';
  const canvases=Array.isArray(model?.active_canvases)?model.active_canvases.map(shape=>Array.isArray(shape)?shape.join('×'):'—').join(', '):'—';
  $('integration-threshold').textContent=model?`${canvases} · ${model.precision||'—'} · порог ${model.threshold}`:'—';
  $('integration-explanations').textContent=health?.explanations?.available?'Доступны':health?.explanations?.reason||'Недоступны';
  $('integration-provider').textContent=health?.explanations?`${health.explanations.provider||'—'} · ${health.explanations.model||'—'}`:'—';
  $('integration-sha').textContent=model?.sha256||'—';
}
$('refresh-service').addEventListener('click',refreshHealth);
async function init(){
  await Promise.all([refreshHistory(),refreshCaseHistory(),refreshHealth()]);
  renderEmpty();
  const page=routePage();history.replaceState(null,'','#'+pageRoutes[page]);showPage(page,{updateHash:false});
  healthTimer=setInterval(refreshHealth,5000);
}
init();

