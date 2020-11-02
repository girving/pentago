// Test machinery for use in either app or actual tests
// Sometimes we want to test in the app to allow GPU debugging

class MidTest {
  let board = try! Board("274440791932540184")
  let board19 = try! Board("686801632813651224")  // slice = 19
  let board20 = try! Board("2609275723747431704")  // slice = 20
  let board22 = try! Board("2060422368541624410")  // slice = 22
  
  let correct = ["274440791932537041":0,"274440791932546937":0,"274440791786788121":1,"274440791950890265":0,"274458384118584601":0,"274463641158554905":0,"684268358023255321":0,"1855204261139584281":1,"274440791932537275":1,"274440791932546235":0,"274440791786788123":0,"274440791950890267":-1,"274458384118584603":0,"274463641158554907":0,"684268358023255323":0,"1855204261139584283":1,"274440791932543593":0,"274440791932546209":0,"274440791786788129":1,"274440791950890273":0,"274458384118584609":0,"274463641158554913":0,"684268358023255329":0,"1855204261139584289":1,"274440791932602568":1,"274440791932611744":0,"274440791787377944":1,"274440791998666008":-1,"274458384118650136":0,"274463641158620440":0,"684268358023320856":0,"1855204261139649816":1,"274440791932537113":1,"274440791932546289":0,"274440791786788201":1,"274440791950890345":0,"274458384118584681":0,"274463641158554985":0,"684268358023255401":0,"1855204261139584361":1,"274440791937845448":1,"274440791937854624":0,"274440791792096536":1,"274440791956198680":0,"274458384123893016":0,"274463641163863320":0,"684268358028563736":0,"1855204261144892696":1,"274440791932537761":0,"274440791932546217":0,"274440791786794681":0,"274440791950896825":-1,"274458384118591161":0,"274463641158561465":0,"684268358023261881":0,"1855204261139590841":0,"274440792075864264":0,"274440792075873440":0,"274440791788557592":0,"274440791966815512":0,"274458384261911832":0,"274463641301882136":0,"684268358166582552":0,"1855204261282911512":0,"274440830587242696":0,"274440830587251872":0,"274440830441493784":0,"274440830605595928":0,"274486563399013656":0,"274463645453522200":0,"684268396677960984":0,"1855204299794289944":0,"274722266909247688":0,"274722266909256864":0,"274722266763498776":0,"274722266927600920":0,"274739859095295256":0,"274745116135265560":0,"686801632813651224":0,"2060399519161652504":0,"276974066722932936":0,"276974066722942112":0,"276974066577184024":0,"276974066741286168":0,"276991658908980504":0,"276996915948950808":0,"2531025680221869336":0,"1855485736116294936":0,"282040616303724744":1,"282040616303733920":0,"282040616157975832":0,"282040616322077976":-1,"282058208489772312":0,"282063465529742616":0,"685112782953387288":0,"2470790035205788952":1,"297240265046100168":-1,"297240265046109344":-1,"297240264900351256":-1,"297240265064453400":-1,"297257857232147736":-1,"297263114272118040":0,"707067831136818456":-1,"1878003734253147416":0,"274443922963695816":1,"274443922963704992":0,"274443922817946904":1,"274443922982049048":0,"274458388413551896":0,"274491820438983960":0,"684271489054414104":0,"1855207392170743064":1,"274450185026013384":-1,"274450185026022560":0,"274450184880264472":-1,"274450185044366616":-1,"274458500082701592":0,"274464684835607832":0,"684277751116731672":0,"1855213654233060632":0,"274468971212966088":1,"274468971212975264":0,"274468971067217176":1,"274468971231319320":0,"274461515149743384":0,"274463679813260568":1,"684296537303684376":0,"1855232440420013336":1,"890026565998741704":0,"890026565998750880":0,"890026565852992792":0,"890026566017094936":0,"890044158184789272":0,"890049415224759576":0,"691868182394443032":0,"1923602680480273688":0,"2121198114131151048":0,"2121198114131160224":0,"2121198113985402136":0,"2121198114149504280":0,"2121215706317198616":0,"2121220963357168920":0,"889463616045323544":0,"1857737535929980184":0,"274440791932540184":1,"274440791932540185m":0,"274440791932540187m":1,"274440791932540193m":0,"274440791932605720m":1,"274440791932540265m":0,"274440791937848600m":0,"274440791932546745m":1,"274440792075867416m":0,"274440830587245848m":0,"274722266909250840m":0,"276974066722936088m":0,"282040616303727896m":1,"297240265046103320m":1,"274443922963698968m":0,"274450185026016536m":1,"274468971212969240m":0,"890026565998744856m":0,"2121198114131154200m":0]
}
