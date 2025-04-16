import Link from "next/link";
import React from "react";
import Image from "next/image";

const Navbar = () => {
  return (
    <div className="flex justify-center w-screen">
      <div className="flex flex-col justify-between items-center w-fit p-2 px-14 bg-blue-500 rounded-b-4xl">
        <Link href="/">
          <Image
            src="/practo_black.jpg"
            width={80}
            height={80}
            className="rounded-full -my-5"
            alt="logo"
          />
        </Link>
        <div>
          <input
            className="bg-white rounded-xl text-black px-3 w-200 h-7"
            type="text"
            placeholder="search"
          />
        </div>
      </div>
    </div>
  );
};

export default Navbar;
