<div class="all">
  <header>
    <h2>Pentago is a first player win</h2>
    <h3>An interactive explorer for perfect pentago play</h3>
  </header>

  <div class="main">
    <nav class="contents">
      <a href="/">Home</a>
      <a href="details.html#intro">Introduction</a>
      <a href="details.html#rules">Pentago rules</a>
      <a href="details.html#algorithms">Algorithms</a>
      <a href="details.html#server">Server setup</a>
      <a href="details.html#open">Code and data</a>
      <a href="details.html#useful">Usefulness</a>
      <a href="details.html#thanks">Acknowledgements</a>
      <a href="details.html#contact">Contact</a>
      {#if window.location.pathname == '/'}
        <a class="back" href="{back}">Back</a>
      {/if}
    </nav>

    <div class="content">
      <nav class="back">
        <a class="back" href="{back}">Back</a>
      </nav>
      <Board bind:back/>
    </div>
  </div>
</div>

<script>
  import Board from './Board.svelte'
  let back = null
</script>
