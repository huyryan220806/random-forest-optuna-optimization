// Editable S4 slides, authored with @oai/artifact-tool from frozen CSV/JSON.
// Copy this file into a private build directory linked to the bundled node_modules.
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile, FileBlob } from '@oai/artifact-tool';

const root = process.env.PROJECT_ROOT;
const skill = process.env.PRESENTATIONS_SKILL_DIR;
const python = process.env.ARTIFACT_PYTHON;
if (!root || !skill || !python) throw new Error('Set PROJECT_ROOT, PRESENTATIONS_SKILL_DIR, ARTIFACT_PYTHON.');
const build = path.join(root, '.build', 's4', 'slides');
await fs.mkdir(build, {recursive:true});
const {finalizePresentation, applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const read = async name => JSON.parse(await fs.readFile(path.join(root,'results',name),'utf8'));
const final = await read('final_metrics.json');
const best = await read('best_config.json');
const b = final.models.baseline, o = final.models.optimized;
// First 9 columns are scalar fields without quoted commas. The trailing fold JSON is unused here.
const lines = (await fs.readFile(path.join(root,'results','trials.csv'),'utf8')).trim().split(/\r?\n/);
// Round chart snapshots to 10 decimal places for Excel's 15 significant-digit limit.
// Full scientific precision remains in the original CSV/JSON.
const rows = lines.slice(1).map(line=>line.split(',').slice(0,9)).map(v=>({trial:+v[0],auc:+Number(v[4]).toFixed(10),sec:+Number(v[6]).toFixed(10),state:v[7]}));
if (rows.length!==30 || rows.some(r=>r.state!=='COMPLETE')) throw new Error('Expected frozen 30 COMPLETE trials.');
const history = rows.map((r,i)=>Math.max(...rows.slice(0,i+1).map(t=>t.auc)));
const pres = Presentation.create({slideSize:{width:1280,height:720}});
const ink='#132D46', teal='#087E8B', muted='#516273', gold='#BD6D20';

function text(slide, value, x,y,w,h,size=28,color=ink,bold=false,name='text') {
  const sh=slide.shapes.add({geometry:'textbox',name,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  sh.text=value;
  sh.text.style={typeface:'Arial',fontSize:size,color,bold,autoFit:'none',wrap:'square',insets:0,verticalAlignment:'top'};
  return sh;
}
function slide(title,notes) {
  const s=pres.slides.add();s.background.fill='#FFFFFF';
  text(s,title,64,46,1152,108,44,ink,true,'primary-heading');
  text(s,'NHẬP MÔN TỐI ƯU HÓA   •   RANDOM FOREST VÀ OPTUNA',64,673,1030,26,16,muted);
  text(s,String(pres.slides.items.length).padStart(2,'0'),1165,670,50,28,18,muted);
  s.speakerNotes.text=notes;
  return s;
}
function table(s,values,y,widths,rowHeight=60,x=64) {
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:widths.reduce((a,b)=>a+b,0),height:rowHeight*values.length,columnWidths:widths,values});
  t.borders.assign({style:'solid',fill:'#D9E0E5',width:1});
  for (let r=0;r<values.length;r++) {
    t.rows[r].height=rowHeight;
    for(let c=0;c<values[r].length;c++){
      const cell=t.getCell(r,c);cell.fill=r===0?ink:r%2?'#F1F5F7':'#FFFFFF';
      cell.text.style={typeface:'Arial',fontSize:23,color:r===0?'#FFFFFF':ink,bold:r===0,verticalAlignment:'middle',insets:12,autoFit:'none'};
    }
  }
  return t;
}
const source='Nguồn số liệu: results/baseline.json, best_config.json, trials.csv, final_metrics.json trong repository. Thực nghiệm chính 25/10/2026, seed 42.';
const repo='https://github.com/huyryan220806/random-forest-optuna-optimization';
const f6=x=>x.toFixed(6);

// 1
{
const s=pres.slides.add();s.background.fill='#FFFFFF';
text(s,'BÀI TẬP LỚN CÁ NHÂN',64,64,1100,38,23,teal,true);
text(s,'Tối ưu Random Forest\nbằng Optuna',64,151,1152,180,64,ink,true,'primary-heading');
text(s,'Phân loại trên Breast Cancer Wisconsin Diagnostic',64,365,1140,52,30,muted);
text(s,'Họ tên: (tự điền)     •     MSSV: (tự điền)\nLớp: (tự điền)     •     Giảng viên: (tự điền)',64,460,1152,92,25,ink);
text(s,'Nhập môn Tối ưu hóa   •   Bản bàn giao 30/10/2026',64,625,1100,34,20,muted);
s.speakerNotes.text='0:00–0:25. Giới thiệu đề tài và câu hỏi: với cùng dữ liệu/folds, TPE có cải thiện CV AUC so với cấu hình mặc định và chi phí bao nhiêu? Đây là benchmark học thuật. Repo: '+repo;
}
// 2
{
const s=slide('Dữ liệu và protocol đánh giá','0:25–1:05. 569 mẫu, 30 đặc trưng; nhãn gốc sklearn malignant=0 được đảo thành malignant=1. Split stratified một lần, train 455 và holdout 114. Chỉ train đi vào CV và tuning. Chốt winner trước test. Nguồn: https://scikit-learn.org/1.7/modules/generated/sklearn.datasets.load_breast_cancer.html ; https://doi.org/10.24432/C5DW2B . '+source);
text(s,'569 mẫu  •  30 đặc trưng  •  malignant = 1',64,162,1152,55,34,teal,true);
table(s,[['Tập dữ liệu','Benign 0','Malignant 1','Tổng'],['Train',285,170,455],['Holdout',72,42,114]],244,[420,244,244,244]);
text(s,'3-fold stratified CV chỉ trên train\nCùng folds, seed 42 và n_jobs=1 cho baseline và mọi trial',64,470,1152,98,29);
text(s,'Chốt tham số bằng CV trước khi đánh giá holdout.',64,593,1152,42,26,teal,true);
}
// 3
{
const s=slide('Mô hình hóa bài toán tối ưu','1:05–1:50. x là ba tham số trong bảng. Ω có 5×7×8=280 cấu hình. Min âm mean AUC tương đương maximize trong code. Baseline mặc định nằm trong miền. 30 trials, 3-fold/trial; baseline ngoài budget search. Thời gian là chi phí đo, không phải ràng buộc cứng. '+source);
text(s,'Mục tiêu   Tối đa hóa ROC-AUC trung bình của 3 folds train',64,156,1152,72,30,teal,true);
table(s,[['Biến quyết định','Miền Ω','Baseline'],['n_estimators','50, 100, 150, 200, 250','100'],['max_depth','None, 4, 6, 8, 12, 16, 24','None'],['min_samples_leaf','Số nguyên 1 đến 8','1']],249,[380,566,206],62);
text(s,'Budget 30 trials  •  Mỗi trial 3 lần fit  •  Không pruning',64,542,1152,48,28);
text(s,'Đo chi phí search, fit và predict; không đặt ngưỡng latency.',64,601,1152,40,25,muted);
}
// 4
{
const s=slide('Baseline và nguyên lý TPE','1:50–2:35. Baseline giữ mọi mặc định trừ seed và n_jobs. TPE phân nhóm lịch sử theo điểm tốt/còn lại, ước lượng l(x), g(x), ưu tiên tỷ lệ cao. Optuna dùng 10 startup trials; smoke 3 trials chưa đi vào giai đoạn khai thác lịch sử. Log cập nhật sau mỗi trial, có state và error; lỗi không gán score giả. Nguồn: https://optuna.readthedocs.io/en/v4.5.0/reference/samplers/generated/optuna.samplers.TPESampler.html ; https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf');
text(s,'Random Forest mặc định',64,175,1100,50,33,teal,true);
text(s,'100 cây, độ sâu không giới hạn, lá tối thiểu 1 mẫu.\nCùng dữ liệu, metric, folds và hàm CV với các trials.',64,236,1152,105,29);
text(s,'Optuna TPESampler',64,369,1100,50,33,teal,true);
text(s,'10 trials ngẫu nhiên khởi tạo, sau đó 20 trials dùng lịch sử.\nƯớc lượng l(x) từ nhóm tốt và g(x) từ nhóm còn lại.\nƯu tiên ứng viên có tỷ lệ l(x)/g(x) cao.',64,432,1152,145,29);
text(s,'CSV/JSON lưu params, điểm từng fold, trạng thái, seed và thời gian.',64,609,1152,36,24,muted);
}
// 5 native line chart
{
const s=slide('Lịch sử điểm CV tốt nhất','2:35–3:20. Đường best-so-far không giảm theo định nghĩa. Best #10 là lần đánh giá 11. Delta CV 0.000395726, khoảng 0.0396 điểm phần trăm. Std 0.003436 baseline và 0.004250 optimized; không phải CI nhiều seed. Trục AUC thu hẹp để đọc chênh lệch nhỏ. '+source);
const c=s.charts.add('line',{position:{left:64,top:176,width:850,height:447},categories:rows.map(r=>String(r.trial+1)),series:[{name:'Best so far',values:history,line:{fill:teal,width:3},marker:{symbol:'none'},smooth:false},{name:'Baseline',values:rows.map(()=>+b.cv_auc_mean.toFixed(10)),line:{fill:gold,width:2},marker:{symbol:'none'},smooth:false}],lineOptions:{smooth:false},hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:{fontSize:20}},xAxis:{title:'Số trial hoàn thành',textStyle:{fontSize:17},numberFormatCode:'0'},yAxis:{title:'CV ROC-AUC',min:0.986,max:0.9905,majorUnit:0.001,numberFormatCode:'0.000',textStyle:{fontSize:18},majorGridlines:{fill:'#D9E0E5',width:1}},chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF'});
applyPresentationChartFont(c,{fontFamily:'Arial'});
text(s,'+0.000396',946,207,270,68,44,teal,true);
text(s,'CV AUC so với\nbaseline',946,291,270,86,27);
text(s,'Best ở lần thử 11\n50 cây / depth 16\nleaf 1',946,427,270,130,27);
text(s,'Trục AUC thu hẹp; mức cải thiện còn nhỏ.',64,630,1152,30,21,muted);
}
// 6 native scatter
{
const s=slide('AUC và chi phí tìm kiếm','3:20–4:00. Một điểm là một trial COMPLETE; x là duration_sec gồm sampling, tạo model, CV và metric logging. Search tổng bao gồm checkpoint CSV, loại trừ baseline CV. Không có latency constraint. Đo thời gian một lần. '+source);
const c=s.charts.add('scatter',{position:{left:64,top:174,width:850,height:449},series:[{name:'Trials',xValues:rows.map(r=>r.sec),values:rows.map(r=>r.auc),marker:{symbol:'circle',size:7,fill:teal},line:{fill:'none',width:0}},{name:'Best #10',xValues:[rows[10].sec],values:[rows[10].auc],marker:{symbol:'diamond',size:12,fill:gold},line:{fill:'none',width:0}}],scatterOptions:{style:'marker'},lineOptions:{smooth:false},hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:{fontSize:20}},xAxis:{title:'Thời gian trial (giây)',numberFormatCode:'0.0',min:0,textStyle:{fontSize:18}},yAxis:{title:'CV ROC-AUC',min:0.975,max:0.995,majorUnit:0.005,numberFormatCode:'0.000',textStyle:{fontSize:18},majorGridlines:{fill:'#D9E0E5',width:1}},chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF'});
applyPresentationChartFont(c,{fontFamily:'Arial'});
text(s,final.search_duration_sec.toFixed(3)+' s',946,206,270,70,44,teal,true);
text(s,'Chi phí search\ntăng thêm',946,291,270,90,27);
text(s,'Baseline CV\n0.716 giây\n\n30 trials COMPLETE',946,423,270,160,26);
text(s,'Thời gian dài hơn không bảo đảm AUC cao hơn; timing chỉ đo một lần.',64,630,1152,30,21,muted);
}
// 7
{
const s=slide('Kết quả trên holdout','4:00–4:45. 114 mẫu test. Winner từ CV, không đổi ngưỡng. AUC tăng nhẹ nhưng các chỉ số phân loại không đổi, do AUC đánh giá thứ tự xác suất còn F1 tại ngưỡng 0.5. Matrix [[72,0],[3,39]] hàng thật, cột dự đoán. Một predict_proba batch 114 mẫu/model. '+source);
table(s,[['Chỉ số','Baseline','Optimized'],['CV AUC',f6(b.cv_auc_mean),f6(o.cv_auc_mean)],['Test AUC',f6(b.test_roc_auc),f6(o.test_roc_auc)],['F1 malignant',f6(b.test_f1),f6(o.test_f1)],['Fit trên train (giây)',b.fit_duration_sec.toFixed(3),o.fit_duration_sec.toFixed(3)],['Predict 114 mẫu (ms)',(b.inference_duration_sec*1000).toFixed(3),(o.inference_duration_sec*1000).toFixed(3)]],170,[560,296,296],60);
text(s,'Cả hai cùng bỏ sót 3 mẫu malignant, không có false positive.',64,566,1152,45,29,teal,true);
text(s,'Ngưỡng cố định 0.5; timing fit/predict chỉ đo một lần.',64,623,1152,35,23,muted);
}
// 8
{
const s=slide('Quyết định và kiểm chứng','4:45–5:20. Giữ optimized theo quy tắc CV trước test, ít cây hơn, chi phí search thêm gần 25 giây. Không tuyên bố ý nghĩa thống kê/hiệu quả lâm sàng. 5:20–6:30 chuyển terminal theo report/DEMO.md: smoke 3, đọc cached final, sinh hình riêng; chỉ ra source code objective không nhận holdout. Chốt kết luận và nguồn. '+source+' Repo: '+repo);
text(s,'Giữ cấu hình 50 cây, depth 16, leaf 1',64,174,1152,60,38,teal,true);
text(s,'CV AUC nhỉnh hơn, model ít cây hơn.\nTest AUC tăng nhẹ; F1 và số mẫu sai không đổi.',64,252,1152,105,30);
text(s,'Giới hạn',64,397,1152,45,32,teal,true);
text(s,'Một dataset, một seed, holdout nhỏ và timing một lần.\nChưa chứng minh ưu thế thống kê hoặc ứng dụng lâm sàng.',64,455,1152,100,29);
text(s,'Demo tiếp theo   Smoke 3 trials, đọc log và sinh lại hình.',64,603,1152,40,26,ink,true);
}

const draft=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(pres)).save(draft);
const finalPath=path.join(root,'report',process.env.SLIDES_FILENAME || 'Trinh_bay_Random_Forest_Optuna.pptx');
await finalizePresentation({
  explicitTotalSlideCount:8,workspaceDir:root,candidatePath:draft,finalPath,
  pythonExecutable:python,
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','2','--require-native-table-slide','3','--require-native-table-slide','7'],
  requiredNativeTableOwnerSlides:[2,3,7],requiredNativeChartOwnerSlides:[5,6],
  materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:['Arial']},
  verifyArtifactToolImport:true,receiptPath:path.join(build,path.basename(finalPath)+'.validation.json')
});
// Preview the finalized bytes, including materialized native chart workbooks.
const verified=await PresentationFile.importPptx(await FileBlob.load(finalPath));
for(const [i,s] of verified.slides.items.entries()){
  const png=await verified.export({slide:s,format:'png',scale:1.5});
  await fs.writeFile(path.join(build,`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
}
console.log(finalPath);
