"""Write the five replay packs in data/replay/ (deterministic; rerun after editing).

    python scripts/build_replay_packs.py

What these packs are, stated plainly:
- svb_2023, deepseek_2025 and tariff_2025 are SYNTHETIC RECONSTRUCTIONS of public events. Each
  headline paraphrases facts that were widely reported at the time and is placed at the
  approximate time it became public. Publishers are ".example" stand-ins (RFC 2606) with a
  credibility tier, so no real outlet is ever quoted with words it did not write. Social posts
  are invented by placeholder accounts. Every record carries "synthetic": true.
- red_team is entirely fictional: "Harbor National Bank" does not exist. It reproduces the
  shape of the May 2023 fake-Pentagon-explosion episode: a lookalike news account, a swarm of
  near-identical reposts from new accounts, then an official denial.
- quiet_day is a fictional calm session used as a control: nothing should trigger.

`python -m seismo live --record ...` produces real packs from GDELT, EDGAR and Bluesky; those
replace the synthetic layer wherever live coverage exists.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "replay"

HEADER = {
    "svb_2023": "SYNTHETIC RECONSTRUCTION of the SVB failure, 8-13 March 2023. Paraphrased headlines timed to the public record; .example publishers; invented social posts.",
    "deepseek_2025": "SYNTHETIC RECONSTRUCTION of the DeepSeek shock, 20-28 January 2025. Paraphrased headlines timed to the public record; .example publishers; invented social posts.",
    "tariff_2025": "SYNTHETIC RECONSTRUCTION of the April 2025 tariff shock, 2-9 April 2025. Paraphrased headlines timed to the public record; .example publishers; invented social posts.",
    "red_team": "FICTIONAL red-team scenario. Harbor National Bank does not exist. A lookalike account, 40 coordinated reposts, then an official denial.",
    "quiet_day": "FICTIONAL calm session used as a control. Nothing in it should trigger a stress test.",
}


def t(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC)


def news(when: str, publisher: str, title: str, body: str = "", **meta) -> dict:
    return {"when": t(when), "type": "news", "publisher": publisher, "title": title, "body": body, "meta": meta}


def filing(when: str, title: str, body: str = "", **meta) -> dict:
    return {"when": t(when), "type": "filing", "publisher": "sec.gov", "title": title, "body": body, "meta": meta}


def post(when: str, author: str, body: str, **meta) -> dict:
    return {"when": t(when), "type": "social", "publisher": "bsky.app", "author": author, "title": "", "body": body, "meta": meta}


# --------------------------------------------------------------------------------------------
# SVB, 8-13 March 2023 (US Eastern = UTC-5 until 12 March, UTC-4 after)
# --------------------------------------------------------------------------------------------
SVB = [
    news("2023-03-08T21:35:00Z", "wire-one.example",
         "SVB Financial plans $2.25 billion capital raise after $1.8 billion loss on securities sale",
         "SVB Financial Group said it sold about $21 billion of securities at a loss of about $1.8 billion and will raise $2.25 billion in new capital to shore up its balance sheet."),
    news("2023-03-09T00:20:00Z", "fin-daily.example",
         "SVB shares tumble after hours on surprise capital raise",
         "Shares of SVB Financial fell sharply in extended trading as investors weighed the loss and the dilution."),
    post("2023-03-09T13:10:00Z", "did:plc:syn-svb-01", "SVB capital raise + securities loss... anyone else nervous about their startup's deposits there?"),
    news("2023-03-09T14:50:00Z", "market-daily.example",
         "SVB Financial shares plunge at the open as liquidity fears mount",
         "SVB Financial stock dropped more than 30% in early trading as investors questioned the lender's liquidity."),
    post("2023-03-09T15:20:00Z", "did:plc:syn-svb-02", "Our VC just told every portfolio company to move their cash out of Silicon Valley Bank today. Feels like a bank run."),
    post("2023-03-09T15:34:00Z", "did:plc:syn-svb-03", "Founders group chat: half of us are wiring money out of SVB right now"),
    post("2023-03-09T15:51:00Z", "did:plc:syn-svb-04", "Pulled our payroll account from SVB this morning. Not waiting to find out."),
    post("2023-03-09T16:05:00Z", "did:plc:syn-svb-05", "If you bank with SVB, talk to your board today. Deposits above the insured limit are at risk."),
    news("2023-03-09T17:05:00Z", "biz-times.example",
         "Venture firms advise startups to withdraw funds from Silicon Valley Bank",
         "Several venture capital firms told portfolio companies to move their deposits out of Silicon Valley Bank as a precaution."),
    post("2023-03-09T17:22:00Z", "did:plc:syn-svb-06", "Every founder I know is trying to withdraw from SVB and the online banking is crawling"),
    post("2023-03-09T17:40:00Z", "did:plc:syn-svb-07", "This is what a bank run looks like in 2023: a group chat and a wire transfer. SVB"),
    news("2023-03-09T18:30:00Z", "street-journal.example",
         "Silicon Valley Bank customers pull deposits as fears of a bank run spread",
         "Customers of Silicon Valley Bank moved deposits to larger lenders, people familiar with the matter said."),
    news("2023-03-09T18:44:00Z", "street-journal-markets.example",
         "SVB deposit outflows accelerate as shares slide about 60%",
         "The sister markets desk reported that Silicon Valley Bank faced heavy deposit withdrawals during the day."),
    news("2023-03-09T21:05:00Z", "wire-two.example",
         "SVB Financial shares close down about 60% in worst day on record",
         "SVB Financial shares lost about 60% of their value. JPMorgan and Bank of America shares also fell as the turmoil hit bank stocks."),
    news("2023-03-09T21:40:00Z", "markets-tv.example",
         "Bank stocks slide as SVB turmoil spreads; JPMorgan and Bank of America fall",
         "Shares of JPMorgan Chase and Bank of America declined as investors worried about contagion from Silicon Valley Bank."),
    news("2023-03-10T13:20:00Z", "wire-one.example",
         "Trading in SVB Financial shares halted before the open",
         "Nasdaq halted trading in SVB Financial shares pending news."),
    news("2023-03-10T16:15:00Z", "regulator.example",
         "Silicon Valley Bank closed by California regulators; FDIC named receiver",
         "Silicon Valley Bank was closed by the California Department of Financial Protection and Innovation, which appointed the FDIC as receiver. Insured depositors will have access to their insured deposits no later than Monday morning."),
    news("2023-03-10T16:40:00Z", "wire-two.example",
         "Silicon Valley Bank collapses in the largest US bank failure since 2008",
         "Silicon Valley Bank failed after customers withdrew about $42 billion in a single day."),
    news("2023-03-10T17:10:00Z", "fin-daily.example",
         "Silicon Valley Bank failure: what happens to uninsured deposits",
         "Most Silicon Valley Bank deposits exceeded the $250,000 insured limit, leaving thousands of companies unsure about payroll."),
    post("2023-03-10T17:30:00Z", "did:plc:syn-svb-08", "SVB is gone. Who's next? First Republic? Signature?"),
    news("2023-03-10T19:30:00Z", "market-daily.example",
         "First Republic shares slide as investors fear contagion from the SVB failure",
         "First Republic Bank stock fell sharply as depositors and investors looked for the next weak lender."),
    news("2023-03-12T22:15:00Z", "central-bank.example",
         "Treasury, Federal Reserve and FDIC say all SVB depositors will be protected; Fed creates emergency lending facility",
         "Regulators invoked a systemic risk exception so depositors of Silicon Valley Bank can access all of their money, and the Federal Reserve announced a new Bank Term Funding Program."),
    news("2023-03-12T22:30:00Z", "wire-one.example",
         "Signature Bank closed by New York regulators in second large bank failure in three days",
         "Signature Bank was closed and depositors will be made whole under the systemic risk exception."),
    news("2023-03-13T12:45:00Z", "market-daily.example",
         "Regional bank shares plunge in premarket despite emergency backstop; First Republic tumbles",
         "First Republic Bank shares fell sharply before the open as investors doubted that the backstop would stop deposit outflows."),
    post("2023-03-13T13:05:00Z", "did:plc:syn-svb-09", "Moving our company accounts from a regional to a big bank this morning. Not taking chances."),
    news("2023-03-13T15:30:00Z", "biz-times.example",
         "JPMorgan and Bank of America see deposit inflows as customers flee regional lenders",
         "Large banks including JPMorgan Chase and Bank of America received a surge of new deposits from customers of smaller banks."),
    news("2023-03-13T20:30:00Z", "wire-two.example",
         "First Republic shares close down about 60% as regional bank rout deepens",
         "First Republic Bank led a broad decline in regional bank stocks."),
]

# --------------------------------------------------------------------------------------------
# DeepSeek, 20-28 January 2025 (US Eastern = UTC-5)
# --------------------------------------------------------------------------------------------
DEEPSEEK = [
    news("2025-01-20T14:00:00Z", "tech-ledger.example",
         "Chinese lab DeepSeek releases R1 reasoning model, says it matches leading US systems",
         "DeepSeek said its open R1 model was trained for a fraction of the cost of US rivals, using far fewer Nvidia chips."),
    news("2025-01-24T16:00:00Z", "tech-ledger.example",
         "DeepSeek's low training cost claims draw attention across the AI industry",
         "Engineers and investors debated whether cheaper training could reduce demand for Nvidia's most expensive chips."),
    post("2025-01-25T15:12:00Z", "did:plc:syn-ds-01", "DeepSeek R1 is free, open and close to o1. What does that do to $NVDA's pricing power?"),
    post("2025-01-25T18:40:00Z", "did:plc:syn-ds-02", "If you can train a frontier model for $6M, the $NVDA capex bull case gets a lot weaker"),
    post("2025-01-25T22:03:00Z", "did:plc:syn-ds-03", "Everyone in my feed is running DeepSeek locally this weekend. Feels like a turning point for $NVDA"),
    post("2025-01-26T02:30:00Z", "did:plc:syn-ds-04", "Hot take: DeepSeek is overhyped, $NVDA still sells every GPU it makes"),
    news("2025-01-26T14:00:00Z", "tech-ledger.example",
         "DeepSeek's assistant climbs to the top of the US App Store download chart",
         "The free DeepSeek app overtook rival chatbots in US downloads over the weekend."),
    post("2025-01-26T16:22:00Z", "did:plc:syn-ds-05", "DeepSeek #1 on the App Store. Monday open for $NVDA is going to be ugly"),
    post("2025-01-26T19:48:00Z", "did:plc:syn-ds-06", "Selling some $NVDA before Monday. A cheaper rival model changes the demand math."),
    news("2025-01-26T21:30:00Z", "fin-daily.example",
         "Investors question AI spending as cheap Chinese model gains traction; Nvidia in focus",
         "Fund managers said the DeepSeek release raised doubts about the hundreds of billions planned for AI data centres, putting Nvidia shares in focus."),
    post("2025-01-27T01:10:00Z", "did:plc:syn-ds-07", "Nasdaq futures sliding tonight. $NVDA premarket is going to be a bloodbath"),
    news("2025-01-27T09:30:00Z", "market-daily.example",
         "Nvidia shares slide in premarket as DeepSeek fears hit the AI trade",
         "Nvidia fell sharply in early premarket trading as a cheaper rival AI model spooked investors."),
    news("2025-01-27T11:05:00Z", "wire-one.example",
         "Tech stocks set for sharp losses as DeepSeek challenges AI spending; Nvidia down more than 10% premarket",
         "Nvidia shares dropped more than 10% before the open as investors reassessed demand for AI chips."),
    post("2025-01-27T11:20:00Z", "did:plc:syn-ds-08", "$NVDA -12% premarket. The DeepSeek weekend was not priced in."),
    news("2025-01-27T12:30:00Z", "street-journal.example",
         "Nvidia set for steep drop as Chinese rival's cheaper AI model rattles investors",
         "A low-cost model from DeepSeek raised questions about the outlook for Nvidia's high-end chips."),
    news("2025-01-27T14:45:00Z", "biz-times.example",
         "Nvidia plunges at the open, dragging chip stocks and the Nasdaq lower",
         "Nvidia shares tumbled about 15% after the open as the selloff spread to other chipmakers."),
    news("2025-01-27T15:30:00Z", "markets-tv.example",
         "Microsoft and Alphabet shares fall as investors reassess AI spending plans",
         "Shares of Microsoft and Alphabet declined as the DeepSeek release raised doubts about the return on AI investment."),
    post("2025-01-27T16:10:00Z", "did:plc:syn-ds-09", "Watching $NVDA lose a few hundred billion before lunch. Wild."),
    news("2025-01-27T21:10:00Z", "wire-two.example",
         "Nvidia loses about $589 billion in market value, the biggest one-day loss for any US company",
         "Nvidia shares closed down about 17% as the DeepSeek release hit the AI trade."),
    news("2025-01-28T14:40:00Z", "market-daily.example",
         "Nvidia shares rebound after record rout as investors buy the dip",
         "Nvidia stock rose in early trading, recovering part of Monday's losses."),
    news("2025-01-28T21:05:00Z", "wire-one.example",
         "Nvidia recovers part of record loss, rising about 9%",
         "Nvidia shares gained about 9% as some investors argued cheaper AI would expand demand over time."),
]

# --------------------------------------------------------------------------------------------
# Tariff shock, 2-9 April 2025 (US Eastern = UTC-4)
# --------------------------------------------------------------------------------------------
TARIFF = [
    news("2025-04-02T20:20:00Z", "wire-one.example",
         "White House announces sweeping reciprocal tariffs, including a 10% baseline on all imports",
         "The administration announced tariffs of 10% on all imports and higher rates for dozens of trading partners, the largest US tariff increase in decades."),
    news("2025-04-02T21:05:00Z", "fin-daily.example",
         "S&P 500 futures tumble after sweeping tariff announcement",
         "Stock futures fell sharply in after-hours trading as investors digested the scale of the new tariffs."),
    post("2025-04-02T21:30:00Z", "did:plc:syn-tf-01", "These tariffs are way bigger than anyone expected. $AAPL builds most iPhones in China."),
    post("2025-04-02T22:15:00Z", "did:plc:syn-tf-02", "Futures down hard. Tomorrow is going to hurt for the S&P 500"),
    news("2025-04-03T12:30:00Z", "market-daily.example",
         "Apple shares slide in premarket on China tariff exposure",
         "Apple fell in premarket trading as investors weighed the cost of tariffs on iPhones assembled in China."),
    news("2025-04-03T13:15:00Z", "street-journal.example",
         "Global stocks sell off as tariffs stoke recession fears",
         "Markets from Tokyo to Frankfurt fell as investors priced the risk of a global trade war. Wall Street was set for a sharp decline."),
    post("2025-04-03T14:05:00Z", "did:plc:syn-tf-03", "Trade war is here. $AAPL $NKE getting crushed at the open"),
    news("2025-04-03T15:20:00Z", "biz-times.example",
         "JPMorgan and Bank of America shares fall sharply on recession fears from tariffs",
         "Bank stocks dropped as investors bet that a trade war would slow growth and raise loan losses."),
    news("2025-04-03T20:10:00Z", "wire-two.example",
         "S&P 500 falls nearly 5% in its worst day since 2020 as tariffs spark a global selloff",
         "The S&P 500 dropped 4.8% and Apple shares fell about 9% as the trade war escalated."),
    news("2025-04-04T11:10:00Z", "wire-one.example",
         "China retaliates with 34% tariffs on all US goods",
         "Beijing announced a 34% tariff on all imports from the United States, escalating the trade war."),
    post("2025-04-04T11:40:00Z", "did:plc:syn-tf-04", "China retaliates with 34% tariffs. This trade war is not ending soon."),
    news("2025-04-04T15:30:00Z", "central-bank.example",
         "Fed chair says tariffs are larger than expected and likely to raise inflation and slow growth",
         "The Federal Reserve chair said the economic effects of the tariffs, including higher inflation and slower growth, are likely to be larger than expected."),
    news("2025-04-04T20:10:00Z", "biz-times.example",
         "S&P 500 slides another 6% as China retaliates; VIX tops 45",
         "The S&P 500 fell 5.97% and the VIX volatility index closed above 45 as the trade war deepened."),
    news("2025-04-04T20:40:00Z", "markets-tv.example",
         "Apple extends losses as tariff rout deepens",
         "Apple shares fell again as investors priced higher costs for products made in China."),
    news("2025-04-07T13:30:00Z", "market-daily.example",
         "Stocks whipsaw as markets struggle to find a floor after tariff selloff",
         "The S&P 500 swung between gains and losses in volatile trading."),
    news("2025-04-09T17:25:00Z", "wire-one.example",
         "White House pauses higher tariffs for 90 days for most countries; stocks soar",
         "The administration paused the higher reciprocal tariffs for 90 days for most trading partners, while keeping the 10% baseline. Stocks surged."),
    post("2025-04-09T17:40:00Z", "did:plc:syn-tf-05", "90 day tariff pause!! S&P 500 ripping higher"),
    news("2025-04-09T20:10:00Z", "wire-two.example",
         "S&P 500 jumps 9.5% in its best day since 2008 after tariff pause",
         "The S&P 500 surged 9.5% and Apple shares rallied about 15% after the tariff pause."),
]

# --------------------------------------------------------------------------------------------
# Red team (fictional): lookalike account -> coordinated reposts -> official denial
# --------------------------------------------------------------------------------------------
REPOST_VARIANTS = [
    "Harbor National Bank just halted withdrawals!! Get your money out now",
    "Harbor National Bank just halted withdrawals! get your money out NOW",
    "BREAKING Harbor National Bank just halted withdrawals. Get your money out now",
    "Harbor National Bank just halted withdrawals!!! get your money out now!!",
]
RED_TEAM = [
    news("2026-09-15T14:00:00Z", "wire-one-alerts.example",
         "BREAKING: Harbor National Bank halts customer withdrawals amid liquidity crisis",
         "Harbor National Bank has halted all customer withdrawals, according to a source, as the lender faces a liquidity crisis.",
         red_team=True),
]
for i in range(40):
    when = datetime(2026, 9, 15, 14, 1, tzinfo=UTC) + timedelta(seconds=25 * i)
    RED_TEAM.append(post(when.isoformat().replace("+00:00", "Z"), f"did:plc:new-acct-{i:02d}",
                         REPOST_VARIANTS[i % len(REPOST_VARIANTS)], red_team=True, account_age_days=1 + i % 4))
RED_TEAM += [
    post("2026-09-15T14:22:00Z", "did:plc:syn-rt-org-01", "Is the Harbor National Bank withdrawal story real? Can't find it on any wire.", red_team=True),
    post("2026-09-15T14:26:00Z", "did:plc:syn-rt-org-02", "My Harbor National app works fine, just moved money between accounts. Smells fake.", red_team=True),
    news("2026-09-15T14:35:00Z", "newsroom.harbornational.example",
         "Harbor National Bank statement on false reports",
         "Reports that Harbor National Bank has halted withdrawals are false. All branches and digital channels are operating normally.",
         official=True, red_team=True),
    news("2026-09-15T14:41:00Z", "wire-one.example",
         "Harbor National Bank denies withdrawal halt; says operations are normal",
         "Harbor National Bank denied a social media report that it had halted withdrawals and said customers can access their money as usual.",
         red_team=True),
]

# --------------------------------------------------------------------------------------------
# Quiet day (fictional control)
# --------------------------------------------------------------------------------------------
QUIET = [
    news("2026-09-16T12:00:00Z", "market-daily.example", "Stock futures little changed ahead of a light data calendar",
         "S&P 500 futures were flat as investors waited for next week's data."),
    news("2026-09-16T12:40:00Z", "fin-daily.example", "Procter & Gamble declares its regular quarterly dividend",
         "Procter & Gamble declared a quarterly dividend in line with its previous payment."),
    news("2026-09-16T13:10:00Z", "biz-times.example", "Caterpillar names new head of its construction unit",
         "Caterpillar said a longtime executive would lead its construction industries segment."),
    post("2026-09-16T13:30:00Z", "did:plc:syn-q-01", "Quiet tape today. $SPY going nowhere"),
    news("2026-09-16T14:20:00Z", "market-daily.example", "Exxon Mobil and Chevron edge higher as oil prices tick up",
         "Energy shares rose modestly as crude oil gained less than 1%."),
    news("2026-09-16T15:00:00Z", "tech-ledger.example", "Microsoft rolls out minor Azure pricing update for storage customers",
         "Microsoft adjusted storage pricing tiers for some Azure customers."),
    post("2026-09-16T15:40:00Z", "did:plc:syn-q-02", "$KO slow and steady as always"),
    news("2026-09-16T16:05:00Z", "money-post.example", "Coca-Cola shares surge to a record high as investors rotate into defensive names",
         "Coca-Cola stock rose about 1% to a record as investors favoured steady consumer staples."),
    news("2026-09-16T16:30:00Z", "fin-daily.example", "Analyst raises price target on Eli Lilly, keeps rating unchanged",
         "An analyst nudged up a price target on Eli Lilly while keeping a neutral rating."),
    news("2026-09-16T17:15:00Z", "money-post.example", "NextEra Energy completes previously announced solar project",
         "NextEra Energy said a solar project announced last year is now operating."),
    news("2026-09-16T18:00:00Z", "market-daily.example", "Treasury yields steady as traders await Fed speakers",
         "The 10-year yield was little changed."),
    post("2026-09-16T18:30:00Z", "did:plc:syn-q-03", "Nothing happening in markets today, perfect day to read 10-Ks"),
    news("2026-09-16T20:10:00Z", "wire-two.example", "Wall Street ends flat in a quiet session",
         "The S&P 500 closed little changed in light volume."),
]

PACKS = {
    "svb_2023": SVB,
    "deepseek_2025": DEEPSEEK,
    "tariff_2025": TARIFF,
    "red_team": RED_TEAM,
    "quiet_day": QUIET,
}


def write_pack(name: str, items: list[dict]) -> Path:
    path = OUT / f"{name}.jsonl"
    items = sorted(items, key=lambda d: d["when"])
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"# {HEADER[name]}\n")
        fh.write("# Built by scripts/build_replay_packs.py. Every record is flagged \"synthetic\": true.\n")
        for i, d in enumerate(items, start=1):
            rec = {
                "doc_id": f"{name}_{i:03d}",
                "source": "replay",
                "source_type": d["type"],
                "publisher": d["publisher"],
                "published_at": d["when"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "title": d["title"],
                "body": d["body"],
                "meta": {"pack": name, **d["meta"]},
                "synthetic": True,
            }
            if d.get("author"):
                rec["author"] = d["author"]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, items in PACKS.items():
        path = write_pack(name, items)
        print(f"{path.relative_to(ROOT)}: {len(items)} documents")


if __name__ == "__main__":
    main()
