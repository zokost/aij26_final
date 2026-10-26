export const fitScale=(width,height,availableWidth,availableHeight)=>Math.min(1,availableWidth/width,availableHeight/height);

export function imagePoint(clientX,clientY,bounds,width,height){
  return [Math.max(0,Math.min(width,(clientX-bounds.left)*width/bounds.width)),
    Math.max(0,Math.min(height,(clientY-bounds.top)*height/bounds.height))];
}

export const dividerPercent=(clientX,bounds)=>Math.max(0,Math.min(100,(clientX-bounds.left)/bounds.width*100));
export const nearDivider=(clientX,bounds,percent,tolerance=12)=>Math.abs(clientX-bounds.left-bounds.width*percent/100)<=tolerance;

export function pointInPolygon([x,y],polygon){
  let inside=false;
  for(let i=0,j=polygon.length-1;i<polygon.length;j=i++){
    const [xi,yi]=polygon[i],[xj,yj]=polygon[j];
    if(Math.abs((x-xi)*(yj-yi)-(y-yi)*(xj-xi))<1e-8&&
      x>=Math.min(xi,xj)&&x<=Math.max(xi,xj)&&y>=Math.min(yi,yj)&&y<=Math.max(yi,yj))return true;
    if((yi>y)!==(yj>y)&&x<(xj-xi)*(y-yi)/(yj-yi)+xi)inside=!inside;
  }
  return inside;
}
