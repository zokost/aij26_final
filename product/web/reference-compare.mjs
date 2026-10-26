import {fitScale} from './viewer.mjs';

const TYPES=['image/jpeg','image/png','image/webp'],MAX_BYTES=20*1024*1024,MAX_PIXELS=15_000_000;

export function validateReferenceFile(file){
  if(!file||!TYPES.includes(file.type))throw new Error('Выберите исходник JPEG, PNG или WebP.');
  if(!Number.isFinite(file.size)||file.size<=0||file.size>MAX_BYTES)throw new Error('Исходник должен быть непустым файлом размером до 20 МБ.');
}

export function referenceSignature(type,bytes){
  if(type==='image/jpeg')return bytes.length>=3&&bytes[0]===255&&bytes[1]===216&&bytes[2]===255;
  if(type==='image/png')return bytes.length>=8&&[137,80,78,71,13,10,26,10].every((b,i)=>bytes[i]===b);
  if(type==='image/webp')return bytes.length>=12&&String.fromCharCode(...bytes.slice(0,4))==='RIFF'&&String.fromCharCode(...bytes.slice(8,12))==='WEBP';
  return false;
}

export function sameDimensions(first,second){return first?.width>0&&first?.height>0&&first.width===second?.width&&first.height===second?.height;}
export function comparisonScale(images,width,height,zoom=1){return Math.min(...images.map(image=>fitScale(image.width,image.height,Math.max(1,Math.floor(width)-1),Math.max(1,Math.floor(height)-1))))*zoom;}
export function scrollFraction(position,contentSize,viewportSize){return contentSize>viewportSize?Math.max(0,Math.min(1,position/(contentSize-viewportSize))):0;}

export function differencePixels(first,second){
  if(!(first instanceof Uint8ClampedArray)||!(second instanceof Uint8ClampedArray)||first.length!==second.length||!first.length||first.length%4)throw new Error('Несовместимые пиксели для сравнения.');
  const result=new Uint8ClampedArray(first.length);
  for(let i=0;i<result.length;i+=4){const value=Math.min(255,4*Math.max(Math.abs(first[i]-second[i]),Math.abs(first[i+1]-second[i+1]),Math.abs(first[i+2]-second[i+2])));result[i]=result[i+1]=result[i+2]=value;result[i+3]=255;}
  return result;
}

export function mountReferenceComparison({host,getCurrent,onChange=()=>{},toast=()=>{}}){
  if(!host||typeof getCurrent!=='function')throw new TypeError('Нужны контейнер и getCurrent для сравнения.');
  host.innerHTML=`<section class="reference-panel" id="reference-panel" hidden>
    <div class="reference-panel-title"><strong>Сравнить с исходником</strong><span>Локально · без нового анализа</span></div>
    <p>Добавьте отдельный исходный кадр. Маска модели и исходник не заменяют друг друга.</p>
    <p id="reference-summary" class="reference-summary" role="status">Исходник ещё не добавлен.</p>
    <div class="reference-actions"><button type="button" class="small-button" id="reference-choose">Добавить исходник</button><button type="button" class="small-button" id="reference-open" disabled>Открыть сравнение</button><button type="button" class="text-button danger" id="reference-remove" hidden>Убрать исходник</button></div>
    <input type="file" id="reference-file" accept="image/jpeg,image/png,image/webp" hidden>
  </section>`;
  const panel=host.querySelector('#reference-panel'),summary=host.querySelector('#reference-summary'),fileInput=host.querySelector('#reference-file'),choose=host.querySelector('#reference-choose'),open=host.querySelector('#reference-open'),remove=host.querySelector('#reference-remove');
  const dialog=document.createElement('dialog');dialog.id='reference-dialog';dialog.className='reference-dialog';
  dialog.setAttribute('aria-labelledby','reference-title');
  dialog.innerHTML=`<div class="reference-dialog-head"><div><strong id="reference-title">Сравнение с исходником</strong><p id="reference-dialog-files"></p></div><button type="button" id="reference-close" class="icon-button" aria-label="Закрыть сравнение с исходником">✕</button></div>
    <div class="reference-toolbar"><label>Вид <select id="reference-view"><option value="pair">Два кадра рядом</option><option value="reference">Исходник</option><option value="current">Проверяемый</option><option value="diff">Различия пикселей</option></select></label><div class="reference-zoom"><button type="button" class="small-button" id="reference-zoom-out" aria-label="Уменьшить оба кадра">−</button><button type="button" class="small-button" id="reference-fit" title="Вписать оба изображения целиком">Вписано</button><button type="button" class="small-button" id="reference-zoom-in" aria-label="Увеличить оба кадра">+</button></div><button type="button" class="small-button" id="reference-blink" aria-pressed="false">Мигание</button></div>
    <p id="reference-note" class="reference-note" role="status"></p>
    <div id="reference-panes" class="reference-panes" hidden>
      <section class="reference-pane" data-reference-pane="reference"><div class="reference-image-title"><strong>ИСХОДНИК</strong><span id="reference-name"></span></div><div class="reference-scroll" id="reference-scroll-original"><div class="reference-content"><img id="reference-image-original" alt="Отдельно приложенный исходник" draggable="false"></div></div></section>
      <section class="reference-pane" data-reference-pane="current"><div class="reference-image-title"><strong>ПРОВЕРЯЕМЫЙ</strong><span id="reference-current-name"></span></div><div class="reference-scroll" id="reference-scroll-current"><div class="reference-content"><img id="reference-image-current" alt="Проверяемое изображение без маски" draggable="false"></div></div></section>
      <section class="reference-pane" data-reference-pane="diff" hidden><div class="reference-image-title"><strong>АБСОЛЮТНАЯ РАЗНОСТЬ RGB ×4</strong><span>Чёрный — совпадение декодированных пикселей</span></div><div class="reference-scroll" id="reference-scroll-diff"><div class="reference-content" id="reference-diff-content"></div></div></section>
    </div><p class="reference-footnote">Совмещение, поиск правки и проверка подлинности исходника не выполняются. Различия могут возникать из-за пересжатия, света, цвета или другого ракурса.</p>`;
  document.body.append(dialog);
  const $=id=>dialog.querySelector('#reference-'+id),panes=$('panes'),viewSelect=$('view'),diffOption=viewSelect.querySelector('option[value="diff"]');
  const scrolls={reference:$('scroll-original'),current:$('scroll-current'),diff:$('scroll-diff')};
  let displayedItem=null,pair=null,operation=null,generation=0,zoom=1,view='pair',blinkTimer=null,syncing=false,layoutWidth=0,layoutHeight=0;
  const eligible=item=>item&&!item.removed&&item.source==='upload'&&item.kind==='image'&&item.file;

  function disposeImage(resource){if(!resource)return;resource.image.src='';URL.revokeObjectURL(resource.url);}
  function cancelOperation(){generation++;operation?.controller.abort();operation=null;}
  function stopBlink(){clearInterval(blinkTimer);blinkTimer=null;$('blink').textContent='Мигание';$('blink').setAttribute('aria-pressed','false');}
  function disposePair(){if(!pair)return;pair.diffController?.abort();disposeImage(pair.reference);disposeImage(pair.current);if(pair.diff){pair.diff.width=pair.diff.height=0;}pair=null;$('image-original').removeAttribute('src');$('image-current').removeAttribute('src');$('diff-content').replaceChildren();panes.hidden=true;}
  function reset(){stopBlink();cancelOperation();disposePair();if(dialog.open)dialog.close();}
  function render(){
    const item=getCurrent();
    if(item!==displayedItem||pair&&(pair.item!==item||pair.referenceFile!==item?.reference?.file||pair.currentFile!==item?.file)){reset();displayedItem=item;}
    panel.hidden=!eligible(item);if(panel.hidden)return;
    const busy=operation?.item===item;choose.disabled=busy;open.disabled=!item.reference||busy;remove.hidden=!item.reference;remove.disabled=busy;
    choose.textContent=item.reference?'Заменить исходник':'Добавить исходник';
    summary.textContent=busy?'Открываем локальные изображения…':item.reference?`${item.reference.file.name} · ${item.reference.width} × ${item.reference.height} px${sameDimensions(item,item.reference)?' · размеры совпадают':' · размеры отличаются, без совмещения'}`:'Исходник ещё не добавлен. JPEG, PNG, WebP · до 20 МБ и 15 млн пикселей.';
  }

  async function decode(file,signal){
    validateReferenceFile(file);
    const bytes=new Uint8Array(await file.slice(0,16).arrayBuffer());signal.throwIfAborted();
    if(!referenceSignature(file.type,bytes))throw new Error('Содержимое исходника не соответствует JPEG, PNG или WebP.');
    const image=new Image(),url=URL.createObjectURL(file),resource={image,url};
    try{
      await new Promise((resolve,reject)=>{const finish=error=>{image.onload=image.onerror=null;signal.removeEventListener('abort',abort);error?reject(error):resolve();};const abort=()=>finish(new DOMException('Ожидание отменено','AbortError'));image.onload=()=>finish();image.onerror=()=>finish(new Error('Не удалось декодировать изображение.'));signal.addEventListener('abort',abort,{once:true});if(signal.aborted){abort();return;}image.src=url;});
      signal.throwIfAborted();
      if(!image.naturalWidth||!image.naturalHeight||image.naturalWidth*image.naturalHeight>MAX_PIXELS)throw new Error('Исходник превышает 15 млн пикселей.');
      return {...resource,width:image.naturalWidth,height:image.naturalHeight};
    }catch(error){disposeImage(resource);throw error;}
  }

  choose.addEventListener('click',()=>fileInput.click());
  fileInput.addEventListener('change',async()=>{
    const file=fileInput.files[0],item=getCurrent();fileInput.value='';if(!file||!eligible(item))return;
    reset();const token=++generation,controller=new AbortController();operation={item,controller,kind:'upload'};render();let decoded;
    try{decoded=await decode(file,controller.signal);if(token!==generation||getCurrent()!==item||item.removed)return;item.reference={file,width:decoded.width,height:decoded.height};onChange(item);toast('Исходник добавлен. Модель его не анализировала.');}
    catch(error){if(error.name!=='AbortError'&&token===generation)toast(error.message);}
    finally{disposeImage(decoded);if(token===generation){operation=null;render();}}
  });
  remove.addEventListener('click',()=>{const item=getCurrent();if(!eligible(item)||!item.reference)return;reset();delete item.reference;onChange(item);render();});

  function setControls(disabled){for(const element of dialog.querySelectorAll('select,button:not(#reference-close)'))element.disabled=disabled;}
  function scrollCenters(){const source=scrolls[view==='pair'?'current':view];return source?[(source.scrollLeft+source.clientWidth/2)/Math.max(1,source.scrollWidth),(source.scrollTop+source.clientHeight/2)/Math.max(1,source.scrollHeight)]:[.5,.5];}
  function layout(resetScroll=false,center=scrollCenters()){
    if(!pair||!dialog.open||panes.hidden)return;
    const visible=Object.entries(scrolls).filter(([name])=>view==='pair'?name!=='diff':name===view),viewport=visible[0]?.[1];if(!viewport?.clientWidth||!viewport.clientHeight)return;
    syncing=true;
    for(const [name,scroll] of Object.entries(scrolls)){
      const image=name==='reference'?pair.reference:pair.current,content=scroll.firstElementChild,scale=comparisonScale([image],viewport.clientWidth,viewport.clientHeight,zoom);
      content.style.width=`${image.width*scale}px`;content.style.height=`${image.height*scale}px`;
      const visual=content.firstElementChild;if(visual){visual.style.width='100%';visual.style.height='100%';}
      if(resetScroll){scroll.scrollLeft=0;scroll.scrollTop=0;}else{scroll.scrollLeft=Math.max(0,center[0]*scroll.scrollWidth-scroll.clientWidth/2);scroll.scrollTop=Math.max(0,center[1]*scroll.scrollHeight-scroll.clientHeight/2);}
    }
    $('fit').textContent=zoom===1?'Вписано':`${Math.round(zoom*100)}%`;$('zoom-out').disabled=zoom<=1;$('zoom-in').disabled=zoom>=4;
    requestAnimationFrame(()=>syncing=false);
  }
  function synchronize(source){
    if(syncing||view!=='pair'||!pair)return;
    const target=source===scrolls.reference?scrolls.current:scrolls.reference,x=scrollFraction(source.scrollLeft,source.scrollWidth,source.clientWidth),y=scrollFraction(source.scrollTop,source.scrollHeight,source.clientHeight),left=x*Math.max(0,target.scrollWidth-target.clientWidth),top=y*Math.max(0,target.scrollHeight-target.clientHeight);
    if(Math.abs(target.scrollLeft-left)<1&&Math.abs(target.scrollTop-top)<1)return;
    syncing=true;target.scrollLeft=left;target.scrollTop=top;requestAnimationFrame(()=>syncing=false);
  }
  for(const scroll of [scrolls.reference,scrolls.current])scroll.addEventListener('scroll',()=>synchronize(scroll));

  async function buildDifference(active){
    const controller=new AbortController();active.diffController=controller;
    const first=document.createElement('canvas'),second=document.createElement('canvas');first.width=second.width=active.current.width;first.height=second.height=active.current.height;
    first.setAttribute('role','img');first.setAttribute('aria-label','Различия декодированных пикселей исходника и проверяемого изображения');
    try{
      const a=first.getContext('2d',{willReadFrequently:true}),b=second.getContext('2d',{willReadFrequently:true});
      for(const [context,image] of [[a,active.current.image],[b,active.reference.image]]){context.fillStyle='#fff';context.fillRect(0,0,first.width,first.height);context.drawImage(image,0,0);}
      const stripe=Math.max(1,Math.floor(512_000/first.width));
      for(let y=0;y<first.height;y+=stripe){controller.signal.throwIfAborted();if(pair!==active)throw new DOMException('Сравнение закрыто','AbortError');const height=Math.min(stripe,first.height-y),current=a.getImageData(0,y,first.width,height),reference=b.getImageData(0,y,first.width,height);current.data.set(differencePixels(current.data,reference.data));a.putImageData(current,0,y);await new Promise(resolve=>requestAnimationFrame(resolve));}
      controller.signal.throwIfAborted();return first;
    }catch(error){first.width=first.height=0;throw error;}
    finally{second.width=second.height=0;active.diffController=null;}
  }
  function note(){return sameDimensions(pair.current,pair.reference)?view==='diff'?'Абсолютная разность RGB ×4, прозрачность сведена на белый фон. Это сравнение пикселей, не вывод модели.':'Одинаковый масштаб и синхронная прокрутка. Автоматическое совмещение не выполнялось.':'Размеры отличаются. Каждый кадр вписан отдельно, увеличение синхронное. Изображения не совмещены; разность пикселей недоступна. Прокрутка — по относительной позиции.';}
  async function setView(next){
    if(!pair)return;const active=pair,center=scrollCenters();
    if(next==='diff'&&!sameDimensions(active.current,active.reference))return;
    view=next;viewSelect.value=view;panes.classList.toggle('reference-single',view!=='pair');for(const section of panes.children)section.hidden=view==='pair'?section.dataset.referencePane==='diff':section.dataset.referencePane!==view;
    $('note').textContent=note();
    if(view==='diff'&&!active.diff){
      if(active.diffController){$('note').textContent='Считаем разность декодированных пикселей локально…';return;}
      $('note').textContent='Считаем разность декодированных пикселей локально…';
      try{const result=await buildDifference(active);if(pair!==active){result.width=result.height=0;return;}active.diff=result;$('diff-content').replaceChildren(result);if(view==='diff')$('note').textContent=note();}
      catch(error){if(error.name!=='AbortError'&&pair===active){toast('Не удалось вычислить разность: '+error.message);if(view==='diff'){view='pair';viewSelect.value=view;panes.classList.remove('reference-single');for(const section of panes.children)section.hidden=section.dataset.referencePane==='diff';layout(false,center);}$('note').textContent=note();}}
    }
    if(pair===active&&view===next)layout(false,center);
  }

  open.addEventListener('click',async()=>{
    const item=getCurrent();if(!eligible(item)||!item.reference)return;reset();
    const token=++generation,controller=new AbortController(),referenceFile=item.reference.file;operation={item,controller,kind:'open'};render();setControls(true);$('note').textContent='Открываем два локальных изображения…';$('dialog-files').textContent=`Исходник: ${referenceFile.name} · Проверяемый: ${item.file.name}`;dialog.showModal();let resources=[];
    try{
      const [referenceResource,currentResource]=await Promise.all([decode(referenceFile,controller.signal).then(value=>{resources.push(value);return value;}),decode(item.file,controller.signal).then(value=>{resources.push(value);return value;})]);
      if(token!==generation||getCurrent()!==item||item.removed||item.reference?.file!==referenceFile)throw new DOMException('Материал сменился','AbortError');
      if(!sameDimensions(referenceResource,item.reference))throw new Error('Размер исходника не совпадает с сохранёнными сведениями.');
      if(currentResource.width!==item.width||currentResource.height!==item.height)throw new Error('Размер проверяемого изображения не совпадает с текущим материалом.');
      pair={item,currentFile:item.file,referenceFile,reference:referenceResource,current:currentResource,diff:null,diffController:null};resources=[];
      $('image-original').src=pair.reference.url;$('image-current').src=pair.current.url;
      $('name').textContent=`${referenceFile.name} · ${pair.reference.width} × ${pair.reference.height} px`;$('current-name').textContent=`${item.file.name} · ${pair.current.width} × ${pair.current.height} px`;
      $('name').title=referenceFile.name;$('current-name').title=item.file.name;
      zoom=1;view='pair';viewSelect.value=view;panes.hidden=false;panes.classList.remove('reference-single');for(const section of panes.children)section.hidden=section.dataset.referencePane==='diff';setControls(false);diffOption.disabled=!sameDimensions(pair.current,pair.reference);$('note').textContent=note();layout(true);
    }catch(error){controller.abort();if(error.name!=='AbortError'&&token===generation){$('note').textContent=error.message;toast(error.message);}}
    finally{for(const resource of resources)disposeImage(resource);if(token===generation){operation=null;render();}}
  });
  viewSelect.addEventListener('change',()=>{stopBlink();setView(viewSelect.value);});
  $('zoom-in').addEventListener('click',()=>{if(pair){const center=scrollCenters();zoom=Math.min(4,zoom+.25);layout(false,center);}});
  $('zoom-out').addEventListener('click',()=>{if(pair){const center=scrollCenters();zoom=Math.max(1,zoom-.25);layout(false,center);}});
  $('fit').addEventListener('click',()=>{zoom=1;layout(true);});
  $('blink').addEventListener('click',()=>{if(!pair)return;if(blinkTimer){stopBlink();return;}setView('current');$('blink').textContent='Остановить мигание';$('blink').setAttribute('aria-pressed','true');blinkTimer=setInterval(()=>{if(!dialog.open||!pair){stopBlink();return;}setView(view==='current'?'reference':'current');},650);});
  $('close').addEventListener('click',()=>dialog.close());
  dialog.addEventListener('close',()=>{if(dialog.open)return;stopBlink();if(pair||operation?.kind==='open'){cancelOperation();disposePair();}render();});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stopBlink();});
  const resize=new ResizeObserver(()=>{const bounds=panes.getBoundingClientRect();if(bounds.width!==layoutWidth||bounds.height!==layoutHeight){layoutWidth=bounds.width;layoutHeight=bounds.height;if(dialog.open&&pair)layout();}});resize.observe(panes);
  render();
  return {render,release(item){if(displayedItem===item||pair?.item===item||operation?.item===item){reset();displayedItem=null;panel.hidden=true;}}};
}
