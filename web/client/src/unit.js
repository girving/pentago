#!/usr/bin/env node
// Unit tests

import { parse_board } from './board.js'
import { strict as assert } from 'assert'
import { readFileSync } from 'fs'
import { midsolve, instantiate } from './mid_sync.js'
import all_games from './games.js'
import pending from './pending.js'
import { get as lru_get, set as lru_set } from './local_lru.js'
const min = Math.min
const pow = Math.pow

async function test_moves() {
  // Check move generation.  Test vectors generated by 'pentago/web/generate moves'
  const moves = {"1011624752379795672":["1011624752379795673m","1011624752379861208m","1011624752379992280m","1011624752380385496m","1011624752379795753m","1011624752379802233m","1011624752523122904m","1011624756674762968m","1011906227356506328m","1014158027170191576m","1011625796056848600m","1011634145473272024m","1011652931660224728m","2858382074578409688m"],"2229312834711074147m":["2229312834711066579","2229312834711066363","2229312834360849763","2229312834332538211","2229333691072263523","2229285827956716899","3307924945466307939","3105262962234635619"],"2410287384723724724":["2410287384723724730m","2410287384724904372m","2410287384723724778m","2410287384723724886m","2410287384723725210m","2410287384723726182m","2410287384723729098m","2410287384819276212m","2415353934304516532m","2410287616651958708m","2455886330950850996m","2410343743284582836m","3641458932856134068m"],"5066549596918038":["5066549596918039m","5066549596918041m","5066549596918047m","5066549596983574m","5066549597507862m","5066549596918065m","5066549596918119m","5066549596918281m","5066549598687510m","5066549602226454m","5066549596918767m","5066549596924599m","5066549644693782m","5066549740245270m","5066550026899734m","5066553891885334m","5066562481819926m","5066588251623702m","5348024573628694m","5910974527050006m","5066665561035030m","5066897489269014m","5067593273970966m","12666373968105750m","27866022710481174m","73464968937607446m","5069680628076822m","5075942690394390m","5094728877347094m","210261807618986262m","620652323663122710m","1851823871795532054m"],"695784702278m":["695784704238","695784703494","695784702278","695784702278","695784702278","695784702278","695784702278","695784702278"],"116043743247":["116043743249m","116043874319m","116044136463m","116044922895m","116043743301m","116043743409m","116043743733m","116047282191m","116054360079m","116043744705m","116043747621m","116043756369m","116330397711m","116903706639m","124633677839m","141813547023m","193353154575m","563065997164559m","1688965904007183m","5066665624535055m","811828445199m","2203397849103m","15199764786118671m","45599062270869519m","136796954725122063m","6378106060815m","18902230695951m","56474604601359m","410390632087879695m","1231171664176152591m","3693514760440971279m"],"6561":["6563m","6567m","6579m","137633m","399777m","1186209m","6615m","6723m","7047m","3545505m","10623393m","31857057m","8019m","10935m","95558049m","286661025m","859969953m","8589941153m","25769810337m","77309417889m","562949953427873m","1688849860270497m","5066549580798369m","231928240545m","695784708513m","2087354112417m","15199648742381985m","45598946227132833m","136796838681385377m","6262062324129m","18786186959265m","56358560864673m","410390516044143009m","1231171548132415905m","3693514644397234593m"],"27m":["3","2187","27","27","27","27","27","27"],
               "5240011396720247894m":["5240011396720250046","5240011396720249110","5240011396352722006","5240011396481696854","5239983221734786134","5240008304343794774","2983707983407629398","3456585944281531478"],"3319782709752515159m":["3319782709752511031","3319782709752515615","3319782709387086423","3319782709420116567","3319759826166762071","3319717013932755543","4008833452740201047","2837897549623872087"],"0":["1m","3m","9m","65536m","196608m","589824m","27m","81m","243m","1769472m","5308416m","15925248m","729m","2187m","6561m","47775744m","143327232m","429981696m","4294967296m","12884901888m","38654705664m","281474976710656m","844424930131968m","2533274790395904m","115964116992m","347892350976m","1043677052928m","7599824371187712m","22799473113563136m","68398419340689408m","3131031158784m","9393093476352m","28179280429056m","205195258022068224m","615585774066204672m","1846757322198614016m"],"4650857499503509072":["4650857499503510530m"],"2977807845258379295":[],"4853554035756516105m":["4853554035756509761","4853554035756517569","4853554036313834249","4853554035787973385","4853528987507245833","4853525895130792713","3070128583317799689","5470547184706274057"],"214483932253520763m":["214483932253520035","214483932253526595","214483932253520763","214483932253520763","214483932253520763","214483932253520763","137922738588222331","2477542745007195003"]}
  for (const name in moves) {
    const board = parse_board(name)
    if (board.name != name)
      throw Error('name inconsistency: '+name+' → '+board.name)
    const correct = moves[name]
    const computed = board.moves().map(b => b.name)
    if (correct+'' != computed+'')
      throw Error('move computation failed: board '+name
        +'\n  correct '+correct.length+' = '+correct
        +'\n  computed '+computed.length+' = '+computed)
  }
}

async function test_done() {
  // Check win testing.  Test vectors generated by 'pentago/web/generate done'
  const done = {"3599295625825626722":1,"4706614627931736233":-1,"2800750759944146110m":0,"3067569452957116051":0,"4686076695755698968m":-1,"0":2,"3455176138216262816":1,"3465003886922835731m":0,"5538920560076335680m":-1,"1564493876535705655m":1,"2864623001501575616m":1,"4915721312399802719":0,"5438451736521550194":0,"2868874237585467318":-1,"3593661990115223505":-1,"2206845117092922089m":2,"244188594m":2,"5463513441749249788":0,"5240293722143534934":-1,"410390593783595011":2,"4693369610226059532":-1,"615585774066204672":2,"2838754993395347559m":1,"2004116360485749876":-1,"3389040007373334375m":-1,"5099274617027833633":-1,"5516118209351010716":0,"68398419340689408m":2,"3693571002959921881":2,"47775744m":2,"3":2,"4898871859185338552":1,"5232699828791233426":0,"5323898997147314514":-1,"4648076302443956741":0,"1200262125340796566":-1,"3687932094289226058m":0,"3419722940059428573m":-1,"3053739602487095877m":0,"4891265683035270920":0,"5241144357739112085":1,"3638365542191859464":2,"5034788427325516227m":0,"5311221911150741914":-1,"4833249440430629392":1,"3218716816241735302":2,"5240272711152123950m":-1,"3616462493856966618":0,"4716747094886727819":1,"3907461234066933715m":2,"7603032727683234m":2,"5304442533049411926m":0,"478002922956790532":-1,"3005642639102068978m":2,"4892932740192151946":0,"5258850059169116008":-1,"3059089035122329738m":2,"3386750347485982824m":1,"3591947619845943885":1,"4900522011443605274":-1,"971146484513188406m":1,"3388451081322574647m":0,"662047931412185088":2,"2055958406107769223":2,"139330113475379929":2,"2178408899810177037m":2,"2872250889127149914m":-1,"4686077494666871049m":-1,"205252660403319618":2,"2289588435192715977m":1,"2846328021321272429m":1,"4715916238222141077m":1,"3067560060194531047":1,"2846355900096850586m":1,"2864903445239712873":1,"5460669340183317367":1,"4776405235193492006":0,"4825659060869022533m":-1,"2795409229536840657":1,"2328117939142132224":-1,"4822301195253662552m":-1,"1331969547228491835m":2,"589824m":2,"4824843927425008714":-1,"2990751626417621906m":0,"5471357043771648583m":-1,"5468823932735926016":-1,"5258008752942760206":0}
  for (const name in done) {
    const board = parse_board(name)
    if (board.name != name)
      throw Error('name inconsistency: name '+name+', board.name '+board.name)
    const v = board.done ? board.immediate_value : 2
    if (v != done[name])
      throw Error('done check failed: board '+name+', correct '+done[name]+', got '+v)
  }
}

async function test_lru() {
  const limit = 10000
  for (let e = 0; e < 2; e++) {
    for (let i = 0; i < limit; i++) {
      if (e == 0 || i != 5)
        assert(lru_get(i) === null)
      lru_set(i, 2*i)
      assert.equal(lru_get(i), 2*i)
    }
    for (let i = 0; i < limit; i++)
      assert.equal(lru_get(i), 2*i)
    lru_set(5, 7) 
    assert.equal(lru_get(5), 7)
    for (let i = 0; i < limit; i++)
      if (i != 5)
        assert(lru_get(i) === null)
  }
}

async function test_wasm() {
  const verbose = false
  const tests = await instantiate(WebAssembly.compile(readFileSync('../build/tests.wasm')))

  // Square test
  const s7 = tests.sqr_test(7)
  assert.equal(s7, 7*7)

  // Die test
  try {
    tests.die_test()
    assert(false)
  } catch (e) {
    assert.equal(e.message, 'An informative message')
  }

  // int64 roundtip test
  for (const n of [0n, 127732n, 144115188075855872n, 5963648625456879657n, 9223372036854775807n])
    for (const x of [n, -n])
      assert.equal(tests.int64_test(n), n)

  // Allocation and sum test
  const data = [1, 0, pow(2, 32) - 1, 7, 3, 13]
  const correct = 7 + 13 + 1
  const ptr = tests.malloc(8 * data.length / 2)
  const chunks = new Uint32Array(tests.memory.buffer, ptr, data.length)
  for (const [i, n] of data.entries())
    chunks[i] = n
  const sum = tests.sum_test(data.length / 2, ptr)
  assert.equal(sum, correct)

  // Big allocation test
  let page = 64 << 10
  let next = tests.malloc(0)
  if (verbose)
    console.log('initial = ' + next)
  const sizes = [0, 33537473, 17, 3537473, 472, 9182]
  for (const size of sizes) {
    const p = tests.malloc(size)
    assert.equal(p, next)
    assert.equal(p % 8, 0)
    next = (p + size + 7) & ~7
    const after = tests.memory.buffer.byteLength
    assert(next <= after)
    assert.equal((next + page - 1) & ~(page - 1), after)
  }
}

async function test_mid() {
  const start = Date.now()
  const board = parse_board('274440791932540184')
  const correct = {'274440791932537041':0,'274440791932546937':0,'274440791786788121':1,'274440791950890265':0,'274458384118584601':0,'274463641158554905':0,'684268358023255321':0,'1855204261139584281':1,'274440791932537275':1,'274440791932546235':0,'274440791786788123':0,'274440791950890267':-1,'274458384118584603':0,'274463641158554907':0,'684268358023255323':0,'1855204261139584283':1,'274440791932543593':0,'274440791932546209':0,'274440791786788129':1,'274440791950890273':0,'274458384118584609':0,'274463641158554913':0,'684268358023255329':0,'1855204261139584289':1,'274440791932602568':1,'274440791932611744':0,'274440791787377944':1,'274440791998666008':-1,'274458384118650136':0,'274463641158620440':0,'684268358023320856':0,'1855204261139649816':1,'274440791932537113':1,'274440791932546289':0,'274440791786788201':1,'274440791950890345':0,'274458384118584681':0,'274463641158554985':0,'684268358023255401':0,'1855204261139584361':1,'274440791937845448':1,'274440791937854624':0,'274440791792096536':1,'274440791956198680':0,'274458384123893016':0,'274463641163863320':0,'684268358028563736':0,'1855204261144892696':1,'274440791932537761':0,'274440791932546217':0,'274440791786794681':0,'274440791950896825':-1,'274458384118591161':0,'274463641158561465':0,'684268358023261881':0,'1855204261139590841':0,'274440792075864264':0,'274440792075873440':0,'274440791788557592':0,'274440791966815512':0,'274458384261911832':0,'274463641301882136':0,'684268358166582552':0,'1855204261282911512':0,'274440830587242696':0,'274440830587251872':0,'274440830441493784':0,'274440830605595928':0,'274486563399013656':0,'274463645453522200':0,'684268396677960984':0,'1855204299794289944':0,'274722266909247688':0,'274722266909256864':0,'274722266763498776':0,'274722266927600920':0,'274739859095295256':0,'274745116135265560':0,'686801632813651224':0,'2060399519161652504':0,'276974066722932936':0,'276974066722942112':0,'276974066577184024':0,'276974066741286168':0,'276991658908980504':0,'276996915948950808':0,'2531025680221869336':0,'1855485736116294936':0,'282040616303724744':1,'282040616303733920':0,'282040616157975832':0,'282040616322077976':-1,'282058208489772312':0,'282063465529742616':0,'685112782953387288':0,'2470790035205788952':1,'297240265046100168':-1,'297240265046109344':-1,'297240264900351256':-1,'297240265064453400':-1,'297257857232147736':-1,'297263114272118040':0,'707067831136818456':-1,'1878003734253147416':0,'274443922963695816':1,'274443922963704992':0,'274443922817946904':1,'274443922982049048':0,'274458388413551896':0,'274491820438983960':0,'684271489054414104':0,'1855207392170743064':1,'274450185026013384':-1,'274450185026022560':0,'274450184880264472':-1,'274450185044366616':-1,'274458500082701592':0,'274464684835607832':0,'684277751116731672':0,'1855213654233060632':0,'274468971212966088':1,'274468971212975264':0,'274468971067217176':1,'274468971231319320':0,'274461515149743384':0,'274463679813260568':1,'684296537303684376':0,'1855232440420013336':1,'890026565998741704':0,'890026565998750880':0,'890026565852992792':0,'890026566017094936':0,'890044158184789272':0,'890049415224759576':0,'691868182394443032':0,'1923602680480273688':0,'2121198114131151048':0,'2121198114131160224':0,'2121198113985402136':0,'2121198114149504280':0,'2121215706317198616':0,'2121220963357168920':0,'889463616045323544':0,'1857737535929980184':0,'274440791932540184':1,'274440791932540185m':0,'274440791932540187m':1,'274440791932540193m':0,'274440791932605720m':1,'274440791932540265m':0,'274440791937848600m':0,'274440791932546745m':1,'274440792075867416m':0,'274440830587245848m':0,'274722266909250840m':0,'276974066722936088m':0,'282040616303727896m':1,'297240265046103320m':1,'274443922963698968m':0,'274450185026016536m':1,'274468971212969240m':0,'890026565998744856m':0,'2121198114131154200m':0}
  assert.equal(Object.keys(correct).length, 1+18+8*18)
  const results = await midsolve(board.raw)
  assert.equal(Object.keys(results).length, 1+18+8*18)
  for (const k of Object.keys(results)) {
    const board = parse_board(k)
    assert(board.name in correct)
    assert.equal(correct[board.name], results[k])
  }
  console.log('    mid time = ' + (Date.now()-start) / 1000 + ' s')
}

async function test_path() {
  const start = Date.now()
  const compute = pending(midsolve)
  await Promise.all(all_games().map(async ({path, values}) => {
    const seen = {}
    await Promise.all(path.map(async name => {
      const board = parse_board(name)
      if (board.count < 18)
        return
      const results = await compute(board.raw + '')
      for (const r in results) {
        const b = parse_board(r).name
        seen[b] = true
        if (b in values && results[r] != values[b])
          throw Error('mismatch: board '+b+', correct '+values[b]+', got '+results[r])
      }
    }))
    for (const b in values) {
      const board = parse_board(b)
      if (!seen[b] && board.count - board.middle >= 18)
        throw Error('missed board ' + b + ', count = ' + board.count)
    }
  }))
  const elapsed = (Date.now() - start) / 1000
  console.log('    path time = ' + elapsed + ' s')
}

const green = '\x1b[1;32m'
const red = '\x1b[1;31m'
const clear = '\x1b[00m'

// Run all tests in series
async function toplevel() {
  let tests = [test_moves, test_done, test_lru, test_wasm, test_mid, test_path]

  // Restrict to specific tests if desired
  if (process.argv.length > 2) {
    const keep = []
    for (const name of process.argv.slice(2)) {
      const test = tests.filter(t => t.name == 'test_' + name)
      if (test.length != 1)
        throw Error('No test named test_' + a + ' found')
      keep.push(test[0])
    }
    tests = keep
  }

  // Launch all tests
  for (const test of tests) {
    try {
      await test()
      console.log('  '+test.name+': '+green+'pass'+clear)
    } catch (e) {
      console.log('  '+test.name+': '+red+'failed'+clear+'\n'+e.stack)
      process.exit(1)
    }
  }
  console.log(green+'  all tests passed'+clear)
  process.exit(0)
}
toplevel()
