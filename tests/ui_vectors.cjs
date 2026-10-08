const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 try{
 const context=await browser.newContext();const page=await context.newPage();page.setDefaultTimeout(90000);
 await page.goto('http://127.0.0.1:8771/');
 await page.locator('#mode').selectOption('qdrant_rerank');await page.locator('#q').fill('Maksekeskus webhook');
 const ready=page.waitForResponse(r=>r.url().includes('mode=qdrant_rerank')&&r.status()===200);
 await page.getByRole('button',{name:'Otsi',exact:true}).click();await ready;
 await page.locator('#results article').first().waitFor();
 const search=await context.request.get('http://127.0.0.1:8771/api/search?q=Maksekeskus%20webhook&mode=qdrant_rerank',{timeout:90000});
 const payload=await search.json();if(payload.results[0].backend!=='qdrant_rerank')throw new Error('Wrong retrieval backend');
 await page.locator('#answerretrieval').selectOption('qdrant_rerank');
 await page.locator('#question').fill('How much does the monthly software subscription cost?');
 await page.locator('#answerbutton').click();
 await page.locator('#answerresult').getByText(/29.00 USD/).waitFor();
 console.log(JSON.stringify({qdrant_reranked_search:true,ui_answer_uses_vector_backend:true,source_amount_verified:true}));
 }finally{await browser.close()}
})().catch(e=>{console.error(e.message);process.exit(1)});
