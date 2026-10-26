const MB=1024*1024,MAX_FRAME_BYTES=50*MB;

export function sampleTimes(duration){
  if(!Number.isFinite(duration)||duration<=0||duration>60)throw new Error('Выберите видео длительностью больше 0 и не более 60 секунд.');
  const count=Math.min(12,Math.max(2,Math.ceil(duration/5)+1)),end=duration-Math.min(.05,duration/2);
  return Array.from({length:count},(_,i)=>i*end/(count-1));
}

const cancelled=()=>new DOMException('Анализ видео отменён.','AbortError');

function timed(action,controller,timeout,message){
  const {signal}=controller;
  return new Promise((resolve,reject)=>{
    let timer;
    const finish=(error,value)=>{clearTimeout(timer);signal.removeEventListener('abort',abort);error?reject(error):resolve(value);};
    const abort=()=>finish(signal.reason||cancelled());
    if(signal.aborted){abort();return;}
    timer=setTimeout(()=>controller.abort(new Error(message)),timeout);
    signal.addEventListener('abort',abort,{once:true});
    Promise.resolve().then(()=>{signal.throwIfAborted();return action(signal);}).then(value=>finish(null,value),finish);
  });
}

function videoReady(video,events,ready,signal,action=()=>{}){
  return new Promise((resolve,reject)=>{
    const finish=error=>{for(const event of events)video.removeEventListener(event,check);video.removeEventListener('error',failed);signal.removeEventListener('abort',abort);error?reject(error):resolve();};
    const check=()=>{if(ready())finish();};
    const failed=()=>finish(new Error('Браузер не смог декодировать видео. Попробуйте другой MP4 или WebM.'));
    const abort=()=>finish(signal.reason||cancelled());
    if(signal.aborted){abort();return;}
    for(const event of events)video.addEventListener(event,check);
    video.addEventListener('error',failed);signal.addEventListener('abort',abort,{once:true});
    try{action();check();}catch(error){finish(error);}
  });
}

/** Explicit scan only. onUpdate is synchronous; analyseFrame(frame, signal) sets frame.analysis. */
export async function analyseVideo(item,{analyseFrame,onUpdate=()=>{},signal}={}){
  if(item.videoScan?.status==='running')throw new Error('Анализ этого видео уже выполняется. Дождитесь завершения или отмените его.');
  const controller=new AbortController(),abort=()=>controller.abort(signal?.reason||cancelled());
  let video,canvas,url,totalBytes=0,watch;
  item.videoFrames=[];item.videoScan={planned_times:[],status:'running'};
  try{
    signal?.addEventListener('abort',abort,{once:true});
    if(signal?.aborted)abort();
    if(item.removed)controller.abort(cancelled());
    controller.signal.throwIfAborted();
    onUpdate(item);
    if(typeof analyseFrame!=='function')throw new Error('Не подключена проверка кадров моделью.');
    if(item.kind!=='video'||!item.file||!['video/mp4','video/webm'].includes(item.file.type)||!item.file.size||item.file.size>100*MB)throw new Error('Выберите MP4 или WebM размером до 100 МБ.');
    if(typeof document==='undefined'||typeof URL.createObjectURL!=='function')throw new Error('Для анализа видео нужен браузер с поддержкой локального видео.');
    watch=setInterval(()=>{if(item.removed)controller.abort(cancelled());},100);
    video=document.createElement('video');video.preload='auto';video.muted=true;video.playsInline=true;
    url=URL.createObjectURL(item.file);
    await timed(s=>videoReady(video,['loadedmetadata'],()=>video.readyState>=1,s,()=>{video.src=url;video.load();}),controller,15000,'Не удалось загрузить параметры видео за 15 секунд.');
    const {videoWidth:width,videoHeight:height,duration}=video;
    if(!Number.isInteger(width)||!Number.isInteger(height)||width<=0||height<=0||width*height>15_000_000)throw new Error('Разрешение видео должно быть не больше 15 млн пикселей.');
    const planned=sampleTimes(duration);
    if(item.width&&item.width!==width||item.height&&item.height!==height||item.duration&&Math.abs(item.duration-duration)>.05)throw new Error('Параметры видео не совпадают с открытым материалом. Откройте файл заново.');
    item.width=width;item.height=height;item.duration=duration;item.videoScan.planned_times=planned;onUpdate(item);
    video.pause();canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;
    const context=canvas.getContext('2d');if(!context)throw new Error('Браузер не поддерживает извлечение кадров видео.');
    for(const target of planned){
      if(item.removed)controller.abort(cancelled());
      controller.signal.throwIfAborted();
      await timed(s=>videoReady(video,['seeked','loadeddata','canplay'],()=>!video.seeking&&video.readyState>=2,s,()=>{video.currentTime=target;}),controller,15000,'Не удалось открыть выбранный кадр за 15 секунд.');
      const timestamp=video.currentTime;
      if(!Number.isFinite(timestamp)||timestamp<0||timestamp>=duration)throw new Error('Браузер не смог выбрать кадр внутри видео.');
      context.drawImage(video,0,0,width,height);
      const blob=await timed(()=>new Promise((resolve,reject)=>canvas.toBlob(value=>value?resolve(value):reject(new Error('Не удалось сохранить кадр PNG.')),'image/png')),controller,15000,'Не удалось подготовить PNG кадра за 15 секунд.');
      if(blob.type!=='image/png'||!blob.size)throw new Error('Браузер не вернул корректный PNG кадра.');
      if(blob.size>20*MB)throw new Error('PNG одного кадра превышает 20 МБ. Частичный результат сохранён.');
      if(totalBytes+blob.size>MAX_FRAME_BYTES)throw new Error('Суммарный размер кадров превысил 50 МБ. Частичный результат сохранён.');
      const filename=`${item.filename} · ${timestamp.toFixed(2)}с.png`;
      const frame={id:crypto.randomUUID(),filename,file:new File([blob],filename,{type:'image/png'}),kind:'image',source:'upload',width,height,regions:[],captured_at:timestamp,parentVideoId:item.id};
      try{
        await timed(s=>analyseFrame(frame,s),controller,120000,'Проверка кадра моделью не завершилась за 120 секунд. Частичный результат сохранён.');
        controller.signal.throwIfAborted();if(item.removed)throw cancelled();
        if(!frame.analysis||typeof frame.analysis.mask_png_base64!=='string'||!frame.analysis.mask_png_base64)throw new Error('Модель не вернула маску кадра. Частичный результат сохранён.');
        item.videoFrames.push(frame);totalBytes+=blob.size;
      }finally{frame.maskImage=null;frame.colorMask=null;}
      onUpdate(item);
    }
    controller.signal.throwIfAborted();if(item.removed)throw cancelled();
    item.videoScan.status='complete';
  }catch(error){
    const wasCancelled=signal?.aborted||item.removed||error?.name==='AbortError';
    item.videoScan.status=wasCancelled?'cancelled':'error';
    if(!wasCancelled)item.videoScan.error=error?.message||'Не удалось выполнить анализ видео. Частичный результат сохранён.';
  }finally{
    clearInterval(watch);signal?.removeEventListener('abort',abort);
    if(video){video.pause();video.removeAttribute('src');video.load();}
    if(url)URL.revokeObjectURL(url);
    if(canvas){canvas.width=0;canvas.height=0;}
    onUpdate(item);
  }
  return item;
}
