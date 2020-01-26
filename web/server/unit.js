#!/usr/bin/env node
// Unit tests

'use strict'
const board_t = require('./board.js')
const Values = require('./values.js')
const Pending = require('./pending.js')
const Log = require('log')
const options = require('commander')
const all_games = require('./games.js')
const block_cache = require('./block_cache.js')
const assert = require('assert').strict
const crypto = require('crypto')

block_cache.set_flaky(0.1)

function test_moves() {
  // Check move generation.  Test vectors generated by 'pentago/web/generate moves'
  const moves = {"1011624752379795672":["1011624752379795673m","1011624752379861208m","1011624752379992280m","1011624752380385496m","1011624752379795753m","1011624752379802233m","1011624752523122904m","1011624756674762968m","1011906227356506328m","1014158027170191576m","1011625796056848600m","1011634145473272024m","1011652931660224728m","2858382074578409688m"],"2229312834711074147m":["2229312834711066579","2229312834711066363","2229312834360849763","2229312834332538211","2229333691072263523","2229285827956716899","3307924945466307939","3105262962234635619"],"2410287384723724724":["2410287384723724730m","2410287384724904372m","2410287384723724778m","2410287384723724886m","2410287384723725210m","2410287384723726182m","2410287384723729098m","2410287384819276212m","2415353934304516532m","2410287616651958708m","2455886330950850996m","2410343743284582836m","3641458932856134068m"],"5066549596918038":["5066549596918039m","5066549596918041m","5066549596918047m","5066549596983574m","5066549597507862m","5066549596918065m","5066549596918119m","5066549596918281m","5066549598687510m","5066549602226454m","5066549596918767m","5066549596924599m","5066549644693782m","5066549740245270m","5066550026899734m","5066553891885334m","5066562481819926m","5066588251623702m","5348024573628694m","5910974527050006m","5066665561035030m","5066897489269014m","5067593273970966m","12666373968105750m","27866022710481174m","73464968937607446m","5069680628076822m","5075942690394390m","5094728877347094m","210261807618986262m","620652323663122710m","1851823871795532054m"],"695784702278m":["695784704238","695784703494","695784702278","695784702278","695784702278","695784702278","695784702278","695784702278"],"116043743247":["116043743249m","116043874319m","116044136463m","116044922895m","116043743301m","116043743409m","116043743733m","116047282191m","116054360079m","116043744705m","116043747621m","116043756369m","116330397711m","116903706639m","124633677839m","141813547023m","193353154575m","563065997164559m","1688965904007183m","5066665624535055m","811828445199m","2203397849103m","15199764786118671m","45599062270869519m","136796954725122063m","6378106060815m","18902230695951m","56474604601359m","410390632087879695m","1231171664176152591m","3693514760440971279m"],"6561":["6563m","6567m","6579m","137633m","399777m","1186209m","6615m","6723m","7047m","3545505m","10623393m","31857057m","8019m","10935m","95558049m","286661025m","859969953m","8589941153m","25769810337m","77309417889m","562949953427873m","1688849860270497m","5066549580798369m","231928240545m","695784708513m","2087354112417m","15199648742381985m","45598946227132833m","136796838681385377m","6262062324129m","18786186959265m","56358560864673m","410390516044143009m","1231171548132415905m","3693514644397234593m"],"27m":["3","2187","27","27","27","27","27","27"],
               "5240011396720247894m":["5240011396720250046","5240011396720249110","5240011396352722006","5240011396481696854","5239983221734786134","5240008304343794774","2983707983407629398","3456585944281531478"],"3319782709752515159m":["3319782709752511031","3319782709752515615","3319782709387086423","3319782709420116567","3319759826166762071","3319717013932755543","4008833452740201047","2837897549623872087"],"0":["1m","3m","9m","65536m","196608m","589824m","27m","81m","243m","1769472m","5308416m","15925248m","729m","2187m","6561m","47775744m","143327232m","429981696m","4294967296m","12884901888m","38654705664m","281474976710656m","844424930131968m","2533274790395904m","115964116992m","347892350976m","1043677052928m","7599824371187712m","22799473113563136m","68398419340689408m","3131031158784m","9393093476352m","28179280429056m","205195258022068224m","615585774066204672m","1846757322198614016m"],"4650857499503509072":["4650857499503510530m"],"2977807845258379295":[],"4853554035756516105m":["4853554035756509761","4853554035756517569","4853554036313834249","4853554035787973385","4853528987507245833","4853525895130792713","3070128583317799689","5470547184706274057"],"214483932253520763m":["214483932253520035","214483932253526595","214483932253520763","214483932253520763","214483932253520763","214483932253520763","137922738588222331","2477542745007195003"]}
  for (const name in moves) {
    const board = new board_t(name)
    if (board.name != name)
      throw Error('name inconsistency: name '+name+', board '+board.name)
    const correct = moves[name]
    const computed = board.moves().map(b => b.name)
    if (correct+'' != computed+'')
      throw Error('move computation failed: board '+name
        +'\n  correct '+correct.length+' = '+correct
        +'\n  computed '+computed.length+' = '+computed)
  }
}

function test_done() {
  // Check win testing.  Test vectors generated by 'pentago/web/generate done'
  const done = {"3599295625825626722":1,"4706614627931736233":-1,"2800750759944146110m":0,"3067569452957116051":0,"4686076695755698968m":-1,"0":2,"3455176138216262816":1,"3465003886922835731m":0,"5538920560076335680m":-1,"1564493876535705655m":1,"2864623001501575616m":1,"4915721312399802719":0,"5438451736521550194":0,"2868874237585467318":-1,"3593661990115223505":-1,"2206845117092922089m":2,"244188594m":2,"5463513441749249788":0,"5240293722143534934":-1,"410390593783595011":2,"4693369610226059532":-1,"615585774066204672":2,"2838754993395347559m":1,"2004116360485749876":-1,"3389040007373334375m":-1,"5099274617027833633":-1,"5516118209351010716":0,"68398419340689408m":2,"3693571002959921881":2,"47775744m":2,"3":2,"4898871859185338552":1,"5232699828791233426":0,"5323898997147314514":-1,"4648076302443956741":0,"1200262125340796566":-1,"3687932094289226058m":0,"3419722940059428573m":-1,"3053739602487095877m":0,"4891265683035270920":0,"5241144357739112085":1,"3638365542191859464":2,"5034788427325516227m":0,"5311221911150741914":-1,"4833249440430629392":1,"3218716816241735302":2,"5240272711152123950m":-1,"3616462493856966618":0,"4716747094886727819":1,"3907461234066933715m":2,"7603032727683234m":2,"5304442533049411926m":0,"478002922956790532":-1,"3005642639102068978m":2,"4892932740192151946":0,"5258850059169116008":-1,"3059089035122329738m":2,"3386750347485982824m":1,"3591947619845943885":1,"4900522011443605274":-1,"971146484513188406m":1,"3388451081322574647m":0,"662047931412185088":2,"2055958406107769223":2,"139330113475379929":2,"2178408899810177037m":2,"2872250889127149914m":-1,"4686077494666871049m":-1,"205252660403319618":2,"2289588435192715977m":1,"2846328021321272429m":1,"4715916238222141077m":1,"3067560060194531047":1,"2846355900096850586m":1,"2864903445239712873":1,"5460669340183317367":1,"4776405235193492006":0,"4825659060869022533m":-1,"2795409229536840657":1,"2328117939142132224":-1,"4822301195253662552m":-1,"1331969547228491835m":2,"589824m":2,"4824843927425008714":-1,"2990751626417621906m":0,"5471357043771648583m":-1,"5468823932735926016":-1,"5258008752942760206":0}
  for (const name in done) {
    const board = new board_t(name)
    const v = board.done() ? board.immediate_value() : 2
    if (v != done[name])
      throw Error('done check failed: board '+name+', correct '+done[name]+', got '+v)
  }
}

function test_str() {
  const s = '1846757322198614016m'
  const b = new board_t(s)
  assert.equal(s, b.name)
}

async function test_pending() {
  const inputs = {0:0, 1:0, 2:0}
  const outputs = {7:0, 8:0, 9:0}
  const slow = Pending(x => new Promise((resolve, reject) => {
    setTimeout(() => { inputs[x]++; resolve(7+x) }, 10) }))
  const s = JSON.stringify
  const expect = e => y => {
    outputs[y]++
    if (y != e) throw Error('bad')
  }
  const all = []
  for (let i = 0; i < 3; i++) all.push(slow(i).then(expect(7+i)))
  for (let i = 0; i < 2; i++) all.push(slow(i).then(expect(7+i)))
  await Promise.all(all)
  if (s(inputs) != s({0:1,1:1,2:1})) throw Error('bad inputs '+s(inputs))
  if (s(outputs) != s({7:2,8:2,9:1})) throw Error('bad outputs '+s(outputs))
}

// Group product for global transforms
const mul = (a, b) => ((a^b)&4)|(((a^(a&b>>2)<<1)+b)&3)

function test_section() {
  // board_section
  const board = new board_t('410395854709526080')
  const section = block_cache.board_section(board)
  assert.deepStrictEqual(section, [[2,1],[1,3],[3,1],[0,1]])

  // transform_section
  const t = block_cache.transform_section
  assert.deepStrictEqual(t(1, section), [[1,3],[0,1],[2,1],[3,1]])
  assert.deepStrictEqual(t(1, t(1, t(1, t(1, section)))), section)
  assert.deepStrictEqual(t(4, section), [[0,1],[1,3],[3,1],[2,1]])
  for (let a = 0; a < 8; a++)
    for (let b = 0; b < 8; b++)
      assert.deepStrictEqual(t(a,t(b,section)), t(mul(a, b), section))

  // standardize_section
  for (let a = 0; a < 8; a++) {
    const [s, b] = block_cache.standardize_section(t(a, section))
    assert.deepStrictEqual(s, section)
    assert.equal(mul(a, b), 0)
  }

  // section_sum, section_shape, block_shape
  assert.equal(block_cache.section_sum(section), 12)
  const shape = [64,126,126,3]
  for (let a = 0; a < 8; a++)
    assert.deepStrictEqual(block_cache.section_shape(t(a, section)), t(a, shape))
  for (let i0 = 0; i0 < 8; i0++)
    for (let i1 = 0; i1 < 16; i1++)
      for (let i2 = 0; i2 < 16; i2++)
        for (let i3 = 0; i3 < 1; i3++)
          assert.deepStrictEqual(block_cache.block_shape(section, [i0,i1,i2,i3]),
                                 [8, i1 < 15 ? 8 : 6, i2 < 15 ? 8 : 6, 3])
}

function test_transform_board() {
  const board = new board_t('410395854709526080')
  const t = (g, b) => new board_t(block_cache.global_transform_board(g, b), 0)
  assert.equal(t(0, board).name, board.name)
  assert.equal(t(1, board).name, '2669284723209146705')
  assert.equal(t(4, board).name, '486429489250764210')
  for (let a = 0; a < 8; a++)
    for (let b = 0; b < 8; b++)
      assert.equal(t(a, t(b, board)).name, t(mul(a, b), board).name)
}

function test_uninterleave() {
  const before = crypto.randomBytes(23 * 64)
  const after = block_cache.uninterleave(before)
  const bit = (d, i) => d[i >> 3] >> (i & 7) & 1
  for (let i = 0; i < 23; i++)
    for (let j = 0; j < 2; j++)
      for (let k = 0; k < 256; k++)
        assert.equal(bit(before, i * 512 + k * 2 + j),
                     bit(after, i * 512 + j * 256 + k))
}

function test_descendent_sections() {
  // Generated from C++ to ensure correct porting
  const known = [
    ['00000000'],
    ['10000000'],
    ['11000000', '01100000', '00011000'],
    ['21000000', '11100000', '01200000', '01101000', '10011000', '00111000', '00012000'],
    ['22000000', '12100000', '02200000', '21010000', '11110000', '02101000', '11011000', '01111000', '10021000',
     '00121000', '01012000', '00022000', '20010100', '10110100', '00210100', '00111100', '01011010', '10010110']
  ]
  const known_counts = [1,1,3,7,18,31,59,101,177,272,427,631,934,1290,1780,2344,3067,3807,4686]
  const known_bits = [0x0,0x1,0x11101,0x31302,0x113102,0x1020123,0x3010211,0x4021220,0x3127061,0x6000000,
                      0x55,0x116225,0x320515,0x11205445,0x10334603,0x139546c,0x232e379,0x78e437,0x32238c37]

  // Compare Javascript and C++
  const name = s => ('' + s).replace(/,/g, '')
  const slices = block_cache.descendent_sections(18)
  assert.equal(slices.length, 18 + 1)
  for (let n = 0; n <= 18; n++) {
    if (n < known.length) {
      assert.equal(known[n].length, slices[n].length)
      for (let i = 0; i < known[n].length; i++)
        assert.equal(known[n][i], name(slices[n][i]))
    }
    assert.equal(known_counts[n], slices[n].length)
    let bits = 0
    for (const s of slices[n])
      bits ^= block_cache.section_sig(s)
    assert.equal(known_bits[n], bits, 'n ' + n + ', known ' + known_bits[n] + ', bits ' + bits)
  }
}

async function test_values() {
  let games = all_games()
  // Truncate
  const truncate = 0
  if (truncate) {
    games = games.slice(0,2)
    for (let i = 0; i < games.length; i++)
      games[i].path = games[i].path.slice(2*19)
  }

  const log = new Log('debug')
  const compute = Values.values(options, log)  // Use a small cache to test replacement
  async function epoch(which) {
    async function test(path, values) {
      const seen = {}
      async function step(name) {
        const board = new board_t(name)
        if (board.count >= 18)
          return
        log.info('board %s launched', name)
        const results = await compute(board)
        for (const b in results) {
          seen[b] = true
          if (b in values && results[b] != values[b])
            throw Error('mismatch: board '+b+', correct '+values[b]+', got '+results[b])
        }
        log.info('board %s checked', name)
      }
      await Promise.all(path.map(step))
      if (!truncate) {
        for (const b in values) {
          const board = new board_t(b)
          if (!seen[b] && board.count - board.middle < 18)
            throw Error('missed board ' + b + ', count = ' + board.count)
        }
      }
    }
    await Promise.all(games.slice(0, which).map(g => test(g.path, g.values)))
  }
  // Run the whole thing twice to make sure caching works
  await epoch(games.length)
  await epoch(1)
}

const green = '\x1b[1;32m'
const red = '\x1b[1;31m'
const clear = '\x1b[00m'

// Parse options
Values.defaults.cache = '4M'  // Use a small cache to test replacement
Values.defaults.bits = 22
Values.defaults.external = true
Values.add_options(options)
options.parse(process.argv)

// Register tests
const tests = [test_moves, test_done, test_pending, test_str, test_section, test_transform_board, test_uninterleave,
               test_descendent_sections]
if (options.args.length > 0) {
  if (options.args.length > 1)
    throw Error('expected 0 or 1 arguments')
  const cmd = options.args[0]
  if (cmd == 'all')
    tests.push(test_values)
  else
    throw Error("unknown command '"+cmd+"'")
}

// Run all tests
Promise.all(tests.map(test =>
  Promise.resolve(test).then(t => t()).then(() =>
    console.log('  '+test.name+': '+green+'pass'+clear)
  ).catch(e => {
    console.log('  '+test.name+': '+red+'failed'+clear+'\n'+e.stack)
    process.exit(1)
  })
)).then(() => {
  console.log(green+'  all tests passed'+clear)
  process.exit(0)
})
