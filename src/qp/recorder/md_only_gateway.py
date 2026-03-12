"""Md-only gateway for Headless Recorder.

目标：
- 无头采集器只连行情服务器（MdApi）
- 绝不连接交易服务器（TdApi）
- 复用 vnpy_ctp / vnpy_tts 已有 MdApi 实现
- 预先注入最小 ContractData，避免 MdApi 因缺失合约缓存丢 tick
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from vnpy.event import Event
from vnpy.trader.constant import Exchange, Product, Status
from vnpy.trader.event import EVENT_TIMER
from vnpy.trader.gateway import BaseGateway
from vnpy.trader.object import (
    CancelRequest,
    ContractData,
    OrderData,
    OrderRequest,
    SubscribeRequest,
)


@dataclass(frozen=True)
class MdOnlyAdapter:
    name: str
    md_api_cls: type
    exchanges: list[Exchange]
    symbol_contract_map: dict
    default_setting: dict
    normalize_address: Callable[[str], str]
    md_connect: Callable[[object, str, str, str, str, dict], None]


def _normalize_tcp_or_ssl(address: str) -> str:
    if not address:
        return address
    if address.startswith(("tcp://", "ssl://", "socks")):
        return address
    return "tcp://" + address


def _build_ctp_adapter() -> MdOnlyAdapter:
    from vnpy_ctp.gateway.ctp_gateway import (
        CtpMdApi,
        CtpGateway,
        EXCHANGE_CTP2VT,
        symbol_contract_map,
    )

    def _connect(md_api, md_address: str, userid: str, password: str, brokerid: str, setting: dict) -> None:
        production_mode: bool = setting.get("柜台环境", "实盘") == "实盘"
        md_api.connect(md_address, userid, password, brokerid, production_mode)

    return MdOnlyAdapter(
        name="CTP",
        md_api_cls=CtpMdApi,
        exchanges=list(EXCHANGE_CTP2VT.values()),
        symbol_contract_map=symbol_contract_map,
        default_setting=CtpGateway.default_setting,
        normalize_address=_normalize_tcp_or_ssl,
        md_connect=_connect,
    )


def _build_tts_adapter() -> MdOnlyAdapter:
    from vnpy_tts.gateway.tts_gateway import (
        TtsMdApi,
        TtsGateway,
        EXCHANGE_TTS2VT,
        symbol_contract_map,
    )

    def _connect(md_api, md_address: str, userid: str, password: str, brokerid: str, setting: dict) -> None:
        md_api.connect(md_address, userid, password, brokerid)

    return MdOnlyAdapter(
        name="TTS",
        md_api_cls=TtsMdApi,
        exchanges=list(EXCHANGE_TTS2VT.values()),
        symbol_contract_map=symbol_contract_map,
        default_setting=TtsGateway.default_setting,
        normalize_address=_normalize_tcp_or_ssl,
        md_connect=_connect,
    )


def get_md_only_gateway_class(gateway_type: str) -> type[BaseGateway]:
    """Factory: build a md-only gateway class for CTP/TTS."""
    gw = gateway_type.upper()
    if gw == "CTP":
        adapter = _build_ctp_adapter()
    elif gw == "TTS":
        adapter = _build_tts_adapter()
    else:
        raise ValueError(f"不支持的 md-only gateway: {gateway_type}")

    class MdOnlyGateway(BaseGateway):
        default_name: str = f"{adapter.name}_MD"
        default_setting: dict = adapter.default_setting
        exchanges: list[Exchange] = adapter.exchanges

        def __init__(self, event_engine, gateway_name: str) -> None:
            super().__init__(event_engine, gateway_name)
            self.adapter = adapter
            self.md_api = adapter.md_api_cls(self)
            self._count = 0
            self._record_symbols: list[str] = []

        def connect(self, setting: dict) -> None:
            userid: str = setting["用户名"]
            password: str = setting["密码"]
            brokerid: str = setting["经纪商代码"]
            md_address: str = adapter.normalize_address(setting["行情服务器"])
            self._record_symbols = list(setting.get("_record_symbols", []))

            self.write_log("无头采集器使用 Md-only 模式：不会连接交易服务器")
            self.write_log(f"将预注册 {len(self._record_symbols)} 个录制合约")

            self._register_stub_contracts(self._record_symbols)
            adapter.md_connect(self.md_api, md_address, userid, password, brokerid, setting)
            self.event_engine.register(EVENT_TIMER, self.process_timer_event)

        def _register_stub_contracts(self, vt_symbols: list[str]) -> None:
            for vt_symbol in vt_symbols:
                symbol, exchange = self._parse_vt_symbol(vt_symbol)
                contract = ContractData(
                    symbol=symbol,
                    exchange=exchange,
                    name=symbol,
                    product=Product.FUTURES,
                    size=1,
                    pricetick=1,
                    gateway_name=self.gateway_name,
                )
                adapter.symbol_contract_map[symbol] = contract
                self.on_contract(contract)
                self.write_log(f"预注册合约: {vt_symbol}")

        def _parse_vt_symbol(self, vt_symbol: str) -> tuple[str, Exchange]:
            if "." not in vt_symbol:
                raise ValueError(f"非法 vt_symbol: {vt_symbol}")
            symbol, exchange_str = vt_symbol.rsplit(".", 1)
            return symbol, Exchange(exchange_str)

        def subscribe(self, req: SubscribeRequest) -> None:
            self.md_api.subscribe(req)

        def close(self) -> None:
            self.event_engine.unregister(EVENT_TIMER, self.process_timer_event)
            self.md_api.close()

        def send_order(self, req: OrderRequest) -> str:
            order = req.create_order_data(
                orderid="MD_ONLY_REJECTED",
                gateway_name=self.gateway_name,
            )
            order.status = Status.REJECTED
            self.on_order(order)
            self.write_log("Md-only gateway 禁止下单")
            return order.vt_orderid

        def cancel_order(self, req: CancelRequest) -> None:
            self.write_log("Md-only gateway 不支持撤单")

        def query_account(self) -> None:
            return

        def query_position(self) -> None:
            return

        def process_timer_event(self, event: Event) -> None:
            self._count += 1
            if self._count < 2:
                return
            self._count = 0
            if hasattr(self.md_api, "update_date"):
                self.md_api.update_date()

    MdOnlyGateway.__name__ = f"{adapter.name}MdOnlyGateway"
    return MdOnlyGateway
