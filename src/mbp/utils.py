def print_section(title, detail, width=60):
    space = width - len(title) - len(detail) - 4
    #if space < 0:
        #raise
        #return
    print("┌" +               "─"*width +             "┐")
    print("│ " + title + ": " + detail + " "*space + " │")
    print("└" +               "─"*width +             "┘")


def print_epoch(epoch, epochs, nsteps, nwalkers, tab_lvl=1, width=60):
    
    epoch = f" Epoch: {epoch:,d} of {epochs:,d}   "
    samples = f" Samples: {nsteps * nwalkers:,d}"
    walkers = f" (Walkers: {nwalkers:,d})"

    width = width - 4 * tab_lvl
    space = width - len(epoch) - len(samples) - len(walkers) - 2
    width = width - len(epoch) - 1

    #if space < 0:
        #raise
        #return
    print(" "*4*tab_lvl + "┌"+"─"*len(epoch)+"┬"+"─"*width+                 "┐")
    print(" "*4*tab_lvl + "│"+epoch+         "│"+samples+" "*space+walkers+" │")
    print(" "*4*tab_lvl + "└"+"─"*len(epoch)+"┴"+"─"*width+                 "┘")


#def print_epoch_stats():

def print_logo(style=1):
    
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    RESET = "\033[0m"  # Reset to default color
   
    COLORS = [RED, GREEN, YELLOW, BLUE]
    RAND = (style + 6145**style) % len(COLORS)
    RANDCOL = COLORS[RAND]
    
    print(" ┌──────┐")
    print("─┤ LOGO ├"+"─"*52+"┐")
    print(" └──────┘")

    if style == 1:
        print(RANDCOL, end="")
        print("             _____ ____________________ ")
        print("            /     \\\\______   \\______   \\")
        print("           /  \\ /  \\|    |  _/|     ___/")
        print("          /    Y    \\    |   \\|    |    ")
        print("          \\____|__  /______  /|____|    ")
        print("                  \\/       \\/           " + RESET)

    elif style == 2:
        print(RANDCOL, end="")
        print("          ┌────────────────────────────────────────┐")
        print("          │         ╭╮            ╭╮             ╭╯│")
        print("          │     ╭───╯│╭╮╭╮  ╭──╮  ││ ─╮        ╭─╯ │")
        print("          │    ╭╯    ││╰╯│  │  ╯  ││╭─╯     ╭──╯   │")
        print("          ─────╯     ╰╯  │  │╭─╯  │╰╯   ╭╮╭─┤      │")
        print("          │              ╰─╮││ ╭──╯╮    │╰╯ ╰──╮   │")
        print("          │                ││╰─╯   ╰────╯      ╰─╮ │")
        print("          │ SS             ╰╯                    ╰╮│")
        print("          └────────────────────────────────────────┘" + RESET)
    elif style == 3:
        print(RANDCOL, end="")
        print("          ┌────────────────────────────────────────┐")
        print("          │ Mo' Bodies, Mo' Problems               │")
        print("          └────────────────────────────────────────┘" + RESET)
    elif style == 4:
        print(RANDCOL, end="")
        print("          ┌────────────────────────────────────────┐")
        print("          │ Mo' Bodies, No Problem                 │")
        print("          └────────────────────────────────────────┘" + RESET)

    else:
        print(RANDCOL, end="")
        print("          ┌────────────────────────────────────────┐")
        print("          │ (M)any (B)ody (P)roblems               │")
        print("          └────────────────────────────────────────┘" + RESET)

    print(" "*53+"┌─────┐")
    print("─"*52+"─┤ ART ├─"+"┘")
    print(" "*53+"└─────┘")
