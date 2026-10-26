// Fixture annotations illustrate the interface; they are not model predictions.
export const operationLabels={addition:'Возможное добавление',removal:'Возможное удаление',copy_move:'Возможное копирование',appearance_change:'Возможное изменение внешнего вида',text_change:'Возможное изменение текста',unknown:'Недостаточно данных'};

export const fixtures = [
  {id:'car', name:'Осмотр автомобиля', filename:'vehicle_inspection.svg', subtitle:'Две области для проверки', src:'assets/car.svg', kind:'image', width:1400, height:900,
    regions:[
      {id:1, title:'Панель двери', hint:'Учебная область на панели двери. Сравните переходы текстуры с соседним участком.', polygon:[[578,455],[701,444],[741,502],[714,557],[587,542],[566,494]]},
      {id:2, title:'Номерной знак', hint:'Небольшая учебная область на номерном знаке. Увеличение помогает рассмотреть детали.', polygon:[[1050,525],[1138,507],[1140,532],[1051,552]]}
    ]},
  {id:'receipt', name:'Мелкая правка', filename:'purchase_receipt.svg', subtitle:'Изменение меньше 1% кадра', src:'assets/receipt.svg', kind:'image', width:1400, height:900,
    regions:[{id:1,title:'Сумма в чеке',hint:'Учебная область пересекается со строкой итоговой суммы. Это демонстрационная разметка; OCR и определение исправленной цифры не выполнялись.',polygon:[[788,572],[839,572],[839,604],[788,604]]}]},
  {id:'clean', name:'Без выделенных областей', filename:'property_inspection.svg', subtitle:'Пример пустой маски', src:'assets/house.svg', kind:'image', width:1400, height:900, regions:[]},
  {id:'video', name:'Видеопроверка', filename:'inspection_storyboard', subtitle:'Таймлайн · 16 учебных кадров', src:'assets/car.svg', kind:'storyboard', width:1400, height:900, duration:15,
    regions:[{id:1,title:'Панель двери',hint:'Учебная область показана в выбранных кадрах сценария. Промежутки между кадрами не проверены моделью.',polygon:[[578,455],[701,444],[741,502],[714,557],[587,542],[566,494]]}]}
];

export function polygonArea(points) {
  return Math.abs(points.reduce((sum,p,i)=>{const q=points[(i+1)%points.length];return sum+p[0]*q[1]-q[0]*p[1];},0))/2;
}

export function enrichRegion(region,width,height) {
  const xs=region.polygon.map(p=>p[0]), ys=region.polygon.map(p=>p[1]);
  const x=Math.min(...xs),y=Math.min(...ys);
  const area_px=polygonArea(region.polygon);
  return {...region,x,y,width:Math.max(...xs)-x,height:Math.max(...ys)-y,area_px,area_frac:area_px/(width*height)};
}

export function frameRegions(item,second=0) {
  const regions=item.kind==='storyboard' && (second<5 || second>10) ? [] : item.regions;
  return [...(item.analysis?.regions||[]),...regions.map(r=>enrichRegion(r,item.width,item.height))];
}

export function modelResult(data) {
  return {...data,regions:data.regions.map((r,i)=>({...r,id:i+1,title:`Область ${i+1}`,annotation_source:'model',
    hint:`Модель выделила ${r.area_px.toLocaleString('ru-RU')} пикселей в этой области. Рассмотрите фрагмент без маски и сравните с окружением. Тип изменения по маске не определяется.`}))};
}

export function statusFor(item,regions) {
  if(item.analysis) return item.analysis.n_regions>0||item.analysis.area_frac>0
    ? {label:'Области для проверки',tone:'attention',detail:'Модель выделила участки с признаками правки. Решение оператора принимается отдельно.'}
    : {label:'Области не выделены',tone:'clear',detail:'При текущем пороге модель не выделила области. Это не подтверждение подлинности.'};
  if(item.source==='upload' && !regions.length) return {label:'Готово к просмотру',tone:'neutral',detail:'Модель не подключена. Автоматической проверки этого файла нет.'};
  if(item.source==='upload') return {label:'Ручные отметки',tone:'attention',detail:'Области отмечены пользователем. Это не вывод модели.'};
  if(regions.length) return {label:'Области для проверки',tone:'attention',detail:'Учебная разметка показывает, как будет представлен результат.'};
  return {label:'Области не выделены',tone:'clear',detail:'Пустая учебная маска. Она не подтверждает подлинность материала.'};
}

export function reportData(item,regions,review,second) {
  const a=item.analysis;
  const frames=(item.videoFrames||[]).map(frame=>reportData(frame,frameRegions(frame),frame.reviewDraft||{decision:'pending',note:''},frame.captured_at));
  return {format_version:1,prototype:!a,generated_at:new Date().toISOString(),
    filename:item.filename,source:item.source==='upload'?'local_upload':'illustrative_fixture',
    analysis_performed:!!a,model:a?.model||null,width:item.width,height:item.height,
    prediction_id:a?.prediction_id||null,input_sha256:a?.input_sha256||null,
    analysed_at:a?.analysed_at||null,inference_ms:a?.inference_ms??null,
    model_mask_area_frac:a?.area_frac??null,model_region_count:a?.n_regions??null,
    regions_truncated:!!a&&a.n_regions>a.regions.length,
    timestamp_seconds:item.captured_at??(['storyboard','video'].includes(item.kind)?second:null),
    annotation_source:a?(item.regions.length?'model_and_manual':'model'):item.source==='upload'?'manual':'illustrative_fixture',
    status:item.kind==='video'?(item.videoScan?.status==='complete'?'Выборка кадров проверена':frames.length?'Видео проверено частично':'Видео не проверено'):statusFor(item,regions).label,regions,review,
    ...(item.reference?{reference:{filename:item.reference.file.name,width:item.reference.width,height:item.reference.height,comparison:'visual_only',aligned:false}}:{}),
    ...(item.kind==='video'?{video:{duration:item.duration,scan:item.videoScan||null,frames,checked_timestamps:frames.map(frame=>frame.timestamp_seconds),coverage:'Проверены только перечисленные моменты. Промежутки между кадрами не проверены.'}}:{}),
    explanations:Object.values(item.explanations||{}),
    external_analysis_performed:Object.keys(item.explanations||{}).length>0,
    limitation:item.kind==='video'?'Проверяются только выбранные кадры. Отсутствие находок на них не подтверждает подлинность всего ролика.':a?'Результат локальной модели. Маска не определяет тип изменения и не удостоверяет подлинность. Оценка вероятности подделки не приводится.':
      'Прототип интерфейса без анализа этого материала. Разметка учебная или ручная; автоматическая проверка не выполнялась.'};
}
